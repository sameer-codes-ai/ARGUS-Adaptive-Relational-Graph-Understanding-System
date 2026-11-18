"""
Baseline 3: Lightweight Robust Temporal Feature Magnitude (RTFM).
Tian et al., ICCV 2021.
Compares against the leading weakly-supervised anomaly paradigm (feature magnitude ranking).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.config import ARGUSConfig
from argus.backbone import SpatialBackbone
from argus.model import ARGUSOutput


class MultiScaleTemporalNetwork(nn.Module):
    """
    Multi-Scale Temporal Network (MTN) using 1D temporal convolutions with
    varying receptive fields (kernels 1, 3, and dilated 3).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        branch_dim = out_dim // 3
        rem_dim = out_dim - 2 * branch_dim

        # Branch 1: Instantaneous point-wise conv
        self.conv1 = nn.Conv1d(in_dim, branch_dim, kernel_size=1, padding=0)

        # Branch 2: Short-term temporal conv (kernel 3)
        self.conv2 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1)

        # Branch 3: Long-term dilated temporal conv (kernel 3, dilation 2)
        self.conv3 = nn.Conv1d(in_dim, rem_dim, kernel_size=3, dilation=2, padding=2)

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) frame features
        Returns:
            out: (B, T, D) multi-scale temporal features
        """
        # (B, T, D) -> (B, D, T) for 1D convolutions
        xt = x.transpose(1, 2)
        b1 = self.conv1(xt)
        b2 = self.conv2(xt)
        b3 = self.conv3(xt)

        fused = torch.cat([b1, b2, b3], dim=1).transpose(1, 2)
        out = self.act(self.norm(fused))
        return out


class RTFMLite(nn.Module):
    """
    Lightweight RTFM Anomaly Detection Architecture.
    Estimates anomaly presence via feature magnitude dynamics in the temporal pyramid.
    """
    def __init__(self, config: Optional[ARGUSConfig] = None):
        super().__init__()
        self.cfg = (config or ARGUSConfig()).validate()

        # Spatial Feature Backbone
        if self.cfg.use_pretrained_backbone:
            self.backbone = SpatialBackbone(
                name=self.cfg.backbone_name,
                out_dim=self.cfg.dim,
                frozen=self.cfg.backbone_frozen,
                pretrained=self.cfg.backbone_pretrained,
                allow_fallback=self.cfg.allow_component_fallbacks,
            )
        else:
            self.backbone = None
            self.patch_proj = nn.Conv2d(
                in_channels=3,
                out_channels=self.cfg.dim,
                kernel_size=self.cfg.patch_size,
                stride=self.cfg.patch_size,
            )

        # Multi-Scale Temporal Network (MTN)
        self.mtn = MultiScaleTemporalNetwork(in_dim=self.cfg.dim, out_dim=self.cfg.dim)

        # Snippet Anomaly Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.dim // 2, 1),
            nn.Sigmoid(),
        )

        # Cache for last computed temporal features (for magnitude loss)
        self.last_temporal_features: Optional[torch.Tensor] = None

    def component_metadata(self) -> dict:
        impl = "patch_projection"
        if self.backbone is not None:
            impl = "timm" if self.backbone.use_timm else "native_convnext"
        return {
            "model_type": "rtfm_lite",
            "backbone_implementation": impl,
            "backbone_name": self.cfg.backbone_name if self.backbone is not None else None,
            "backbone_pretrained_requested": self.cfg.backbone_pretrained,
            "temporal_aggregation": "mtn_multiscale",
        }

    def compute_magnitudes(self, temporal_feats: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of temporal features along channel dimension."""
        return torch.norm(temporal_feats, p=2, dim=-1)  # (B, T)

    def forward(
        self,
        frames: torch.Tensor,
        stream: bool = False,
        stream_id: str = "default",
    ) -> ARGUSOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape (B, T, 3, H, W)")
        b, t, c, h, w = frames.shape
        if b == 0 or t == 0:
            raise ValueError("frames must contain at least one batch item and one frame")

        flat_frames = frames.view(b * t, c, h, w)

        if self.backbone is not None:
            tokens, _ = self.backbone(flat_frames)
            frame_feats = tokens.mean(dim=1)
        else:
            p = self.patch_proj(flat_frames)
            frame_feats = p.mean(dim=(2, 3))

        frame_feats = frame_feats.view(b, t, self.cfg.dim)

        # Multi-scale temporal aggregation
        temporal_feats = self.mtn(frame_feats)
        self.last_temporal_features = temporal_feats if torch.is_grad_enabled() else None

        # Compute anomaly scores
        scores = self.classifier(temporal_feats).squeeze(-1)  # (B, T)

        valid_mask = torch.ones((b, t), dtype=torch.bool, device=frames.device)
        selected_batches = [torch.arange(t, device=frames.device, dtype=torch.long) for _ in range(b)]
        empty_boxes = [[torch.empty((0, 4), device=frames.device) for _ in range(t)] for _ in range(b)]

        return ARGUSOutput(
            scores=scores,
            sampled_indices=selected_batches,
            boxes=empty_boxes,
            valid_mask=valid_mask,
            sequence_lengths=torch.full((b,), t, dtype=torch.long, device=frames.device),
        )
