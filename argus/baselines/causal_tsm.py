"""
Baseline 2: Causal Streaming Temporal Shift Module (TSM).
Provides a direct, streaming causal video benchmark without graph dependencies.
Uses past-only channel shift (+1 time step) with zero future lookahead.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.config import ARGUSConfig
from argus.backbone import SpatialBackbone
from argus.model import ARGUSOutput


class CausalTemporalShift(nn.Module):
    """
    In-place causal temporal channel shift module.
    Shifts the first (1/shift_div) fraction of channels forward by +1 time step
    from past to present. Zero channels are shifted from future to present,
    maintaining strict causal invariance.
    """
    def __init__(self, channels: int, shift_div: int = 8):
        super().__init__()
        self.channels = channels
        self.shift_div = shift_div
        self.shift_channels = max(1, channels // shift_div)

    def forward(
        self,
        x: torch.Tensor,
        buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) feature tensor.
            buffer: Optional (B, 1, C) feature of the last frame from previous clip.
        Returns:
            out: (B, T, C) causally shifted tensor.
            new_buffer: (B, 1, C) last frame feature to cache for next streaming window.
        """
        b, t, c = x.shape
        new_buffer = x[:, -1:].detach()

        # Split into shifted (past) channels and identity channels
        shift_dim = self.shift_channels
        x_shift = x[:, :, :shift_dim]
        x_keep = x[:, :, shift_dim:]

        if buffer is not None and buffer.shape[0] == b:
            # Shift forward using buffer for the first frame
            buf = buffer.to(device=x.device, dtype=x.dtype)
            past_part = torch.cat([buf[:, :, :shift_dim], x_shift[:, :-1]], dim=1)
        else:
            # Zero-padding for the first frame when no buffer is available
            zero_pad = torch.zeros((b, 1, shift_dim), dtype=x.dtype, device=x.device)
            past_part = torch.cat([zero_pad, x_shift[:, :-1]], dim=1)

        out = torch.cat([past_part, x_keep], dim=-1)
        return out, new_buffer


class CausalTSM(nn.Module):
    """
    Causal Streaming ConvNeXt-TSM Video Anomaly Detection Network.
    Computes causal temporal representations without graph overhead.
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

        # Causal Temporal Shift Module (1/8th channel past shift)
        self.tsm = CausalTemporalShift(channels=self.cfg.dim, shift_div=8)

        # Causal Temporal Conv Block
        self.temporal_block = nn.Sequential(
            nn.LayerNorm(self.cfg.dim),
            nn.Linear(self.cfg.dim, self.cfg.dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Output Scoring Head
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.dim // 2, 1),
            nn.Sigmoid(),
        )

        # Streaming Memory Cache: stream_id -> (1, 1, C)
        self._max_streams = 256
        self._stream_buffers: Dict[str, torch.Tensor] = {}

    def reset(self, stream_id: Optional[str] = None):
        """Reset streaming cache."""
        if stream_id is not None:
            self._stream_buffers.pop(str(stream_id), None)
        else:
            self._stream_buffers.clear()

    def component_metadata(self) -> dict:
        impl = "patch_projection"
        if self.backbone is not None:
            impl = "timm" if self.backbone.use_timm else "native_convnext"
        return {
            "model_type": "causal_tsm",
            "backbone_implementation": impl,
            "backbone_name": self.cfg.backbone_name if self.backbone is not None else None,
            "backbone_pretrained_requested": self.cfg.backbone_pretrained,
            "shift_proportion": "1/8_past_only",
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
        if b == 0 or t == 0:
            raise ValueError("frames must contain at least one batch item and one frame")
        if stream and b != 1:
            raise ValueError("streaming inference requires batch size 1")

        flat_frames = frames.view(b * t, c, h, w)

        if self.backbone is not None:
            tokens, _ = self.backbone(flat_frames)
            frame_feats = tokens.mean(dim=1)
        else:
            p = self.patch_proj(flat_frames)
            frame_feats = p.mean(dim=(2, 3))

        frame_feats = frame_feats.view(b, t, self.cfg.dim)

        # Retrieve streaming buffer if stream mode is active
        stream_key = str(stream_id)
        buf = self._stream_buffers.get(stream_key) if stream else None

        # Apply causal temporal shift
        shifted, new_buf = self.tsm(frame_feats, buffer=buf)
        if stream:
            if stream_key not in self._stream_buffers and len(self._stream_buffers) >= self._max_streams:
                self._stream_buffers.pop(next(iter(self._stream_buffers)))
            self._stream_buffers[stream_key] = new_buf

        # Causal residual temporal aggregation
        temporal_feats = frame_feats + self.temporal_block(shifted)

        # Frame anomaly scores
        scores = self.head(temporal_feats).squeeze(-1)  # (B, T)

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
