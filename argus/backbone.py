"""
Spatial Feature Backbone — Pre-trained and Native ConvNeXt dense visual token extractor.
Replaces the naive Conv2D patch projector with rich multi-scale ConvNeXt features.

Implements Module 2 of the ARGUS research design:
  Raw Frame -> Backbone (ConvNeXt / ViT) -> Dense Visual Tokens (B, N_patches, D) + Multi-Scale Maps
"""

from __future__ import annotations
import math
import warnings
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Native ConvNeXt Block & Stage Implementation (Pure PyTorch) ──────────

class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with depthwise 7x7 conv, inverted bottleneck (4x expansion),
    LayerNorm, GELU, and layer scale.
    """
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, 1, 1)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input_tensor + x


class NativeConvNeXt(nn.Module):
    """
    Pure-PyTorch ConvNeXt architecture (matches convnext_tiny layout).
    Stages: [3, 3, 9, 3] blocks with dims [96, 192, 384, 768].
    Stride 16 at Stage 2 / Stride 32 at Stage 3.
    """
    def __init__(self, in_chans: int = 3, depths: List[int] = [2, 2, 6, 2],
                 dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: 4x4 stride-4 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Downsampling stages (stride 2 between stages)
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.channel_dims = dims

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_maps = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_maps.append(x)
        return feature_maps


# ── Unified Spatial Backbone (timm with Native Fallback) ───────────────────

class SpatialBackbone(nn.Module):
    """
    Spatial Feature Backbone for dense patch and RoI feature extraction.
    Uses ``timm`` when available, or native ConvNeXt when timm is not installed.
    """
    def __init__(
        self,
        name: str = "convnext_tiny",
        out_dim: int = 256,
        frozen: bool = True,
        pretrained: bool = True,
        allow_fallback: bool = True,
    ):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.use_timm = False
        self._frozen = False

        self.pretrained_requested = pretrained
        self.fallback_reason: Optional[str] = None

        try:
            if name in {"native", "native_convnext", "native_convnext_tiny"}:
                raise ImportError("native backbone explicitly selected")
            import timm
            self.model = timm.create_model(name, pretrained=pretrained, features_only=True)
            self.use_timm = True
            self.model.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 128, 128)
                feats = self.model(dummy)
                self.stage_dims = [f.shape[1] for f in feats]
                self.stage_strides = [128 // max(f.shape[2], 1) for f in feats]
            if not frozen:
                self.model.train()
            self.token_stage_idx = len(self.stage_dims) - 1
            for i, s in enumerate(self.stage_strides):
                if s == 16:
                    self.token_stage_idx = i
                    break
            self.feat_dim = self.stage_dims[self.token_stage_idx]
        except Exception as exc:
            explicit_native = name in {"native", "native_convnext", "native_convnext_tiny"}
            if not explicit_native and not allow_fallback:
                raise RuntimeError(
                    f"Unable to initialize requested backbone '{name}'. Install timm and make "
                    "the pretrained weights available, or explicitly enable component fallbacks."
                ) from exc
            # Native ConvNeXt fallback
            self.fallback_reason = str(exc)
            if not explicit_native:
                warnings.warn(
                    f"Falling back to a randomly initialized native ConvNeXt because '{name}' "
                    f"could not be loaded: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.use_timm = False
            self.model = NativeConvNeXt(depths=[2, 2, 4, 2], dims=[64, 128, 256, 384])
            # Stage 2 has stride 16 (4 * 2 * 2 = 16)
            self.token_stage_idx = 2
            self.feat_dim = self.model.channel_dims[self.token_stage_idx]
            self.stage_dims = self.model.channel_dims

        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, out_dim),
        )

        if frozen and self.fallback_reason is not None and pretrained:
            raise RuntimeError(
                "Refusing to freeze a random fallback backbone. Select an explicit native backbone "
                "with backbone_pretrained=false, or leave it trainable."
            )

        if frozen:
            self.freeze()

    def freeze(self):
        """Freeze backbone weights so only token projection head trains."""
        self._frozen = True
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze backbone for end-to-end fine-tuning."""
        self._frozen = False
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 3, H, W) or (3, H, W) image tensor.
        Returns:
            visual_tokens: (B, N_patches, out_dim)
            feature_maps: list of feature maps from each stage
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.use_timm:
            feature_maps = self.model(x)
        else:
            feature_maps = self.model.forward_features(x)

        feat = feature_maps[self.token_stage_idx]  # (B, C, H', W')
        tokens = feat.flatten(2).transpose(1, 2)   # (B, H'*W', C)
        visual_tokens = self.token_proj(tokens)     # (B, N_patches, out_dim)

        return visual_tokens, feature_maps
