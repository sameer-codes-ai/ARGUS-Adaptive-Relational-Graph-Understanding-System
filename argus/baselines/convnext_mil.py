"""
Baseline 1: ConvNeXt-Tiny + Temporal Pooling + Multiple Instance Learning (MIL).
Quantifies the exact value added by ARGUS's scene graph and EdgeGAT over a standard 2D CNN.
"""

from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.config import ARGUSConfig
from argus.backbone import SpatialBackbone
from argus.model import ARGUSOutput


class ConvNeXtMIL(nn.Module):
    """
    Standard Weakly-Supervised 2D CNN baseline.
    Extracts dense frame features via ConvNeXt-Tiny, pools spatial features,
    and classifies snippets with a standard temporal projection head.
    """
    def __init__(self, config: Optional[ARGUSConfig] = None):
        super().__init__()
        self.cfg = (config or ARGUSConfig()).validate()

        # Spatial Backbone (ConvNeXt-Tiny, compute-matched to ARGUS)
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

        # Snippet Anomaly Scoring Head
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.dim // 2, 1),
            nn.Sigmoid(),
        )

    def component_metadata(self) -> dict:
        impl = "patch_projection"
        if self.backbone is not None:
            impl = "timm" if self.backbone.use_timm else "native_convnext"
        return {
            "model_type": "convnext_mil",
            "backbone_implementation": impl,
            "backbone_name": self.cfg.backbone_name if self.backbone is not None else None,
            "backbone_pretrained_requested": self.cfg.backbone_pretrained,
        }

    def forward(
        self,
        frames: torch.Tensor,
        stream: bool = False,
        stream_id: str = "default",
    ) -> ARGUSOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape (B, T, 3, H, W)")
        b, t, c, h, w = frames.shape

        # Reshape to (B * T, 3, H, W) for spatial feature extraction
        flat_frames = frames.view(b * t, c, h, w)

        if self.backbone is not None:
            tokens, _ = self.backbone(flat_frames)  # (B*T, N_patches, dim)
            frame_feats = tokens.mean(dim=1)        # (B*T, dim)
        else:
            p = self.patch_proj(flat_frames)        # (B*T, dim, H', W')
            frame_feats = p.mean(dim=(2, 3))        # (B*T, dim)

        frame_feats = frame_feats.view(b, t, self.cfg.dim)

        # Snippet anomaly scores in [0, 1]
        scores = self.head(frame_feats).squeeze(-1)  # (B, T)

        valid_mask = torch.ones((b, t), dtype=torch.bool, device=frames.device)
        selected_batches = [list(range(t)) for _ in range(b)]
        empty_boxes = [[torch.empty((0, 4), device=frames.device) for _ in range(t)] for _ in range(b)]

        return ARGUSOutput(
            scores=scores,
            sampled_indices=selected_batches,
            boxes=empty_boxes,
            valid_mask=valid_mask,
            sequence_lengths=torch.full((b,), t, dtype=torch.long, device=frames.device),
        )
