"""
Adaptive Frame-Sampling Controller.

Implements Module 1 of the ARGUS research design:
  - Dynamically selects informative keyframes from video streams.
  - Combines motion magnitude (velocity), gradient energy, histogram-based temporal entropy,
    and frame differencing.
  - Guarantees minimum temporal coverage via max_frame_gap.
  - Computes dynamic attention modulation factors for the temporal transformer based on
    spatial movement velocity and temporal entropy.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSampler(nn.Module):
    def __init__(self, threshold: float = 0.12, max_gap: int = 4, uniform_sampling: bool = False):
        super().__init__()
        self.threshold = threshold
        self.max_gap = max_gap
        self.uniform_sampling = uniform_sampling

    def importance_scores(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[:2]
        gray = frames.mean(dim=2)
        if t <= 1:
            return torch.ones((b, t), device=frames.device)

        # Spatial movement velocity proxy (inter-frame diff)
        diff = (gray[:, 1:] - gray[:, :-1]).abs().mean(dim=(-1, -2))
        grad_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(-1, -2))
        grad_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=(-1, -2))
        spatial_energy = (grad_x + grad_y)[:, 1:]

        # Histogram-based temporal entropy (32-bin Shannon entropy)
        # Accumulate in float32 to prevent float16 overflow when H*W > 65504.
        bins = 32
        quantized = (gray[:, 1:] * (bins - 1)).long().clamp(0, bins - 1)
        flat_q = quantized.flatten(2)  # (B, T-1, H*W)
        hist = torch.zeros(b, t - 1, bins, device=frames.device, dtype=torch.float32)
        hist.scatter_add_(2, flat_q, torch.ones_like(flat_q, dtype=torch.float32))
        p = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)  # (B, T-1)
        entropy_std = entropy.std(dim=-1, keepdim=True, unbiased=False)
        entropy = ((entropy - entropy.mean(dim=-1, keepdim=True)) / (entropy_std + 1e-6)).to(dtype=frames.dtype)

        hist_diff = (gray[:, 1:] - gray[:, :-1]).pow(2).mean(dim=(-1, -2))

        step_score = diff + 0.3 * spatial_energy + 0.1 * F.relu(entropy) + 0.2 * hist_diff
        first = step_score[:, :1]
        scores = torch.cat([first, step_score], dim=1)
        return scores

    def select(self, frames: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        b, t = frames.shape[:2]
        if self.uniform_sampling:
            selected_batches = [torch.arange(t, device=frames.device, dtype=torch.long) for _ in range(b)]
            scores = torch.ones((b, t), device=frames.device, dtype=frames.dtype)
            return selected_batches, scores
        scores = self.importance_scores(frames)
        above_thresh = (scores >= self.threshold).cpu().tolist()
        selected_batches = []
        for i in range(b):
            chosen = [0]
            last = 0
            row = above_thresh[i]
            for k in range(1, t):
                gap = k - last
                if row[k] or gap >= self.max_gap or k == t - 1:
                    chosen.append(k)
                    last = k
            selected_batches.append(torch.tensor(chosen, device=frames.device, dtype=torch.long))
        return selected_batches, scores

    def compute_attention_modulation(
        self,
        scores: torch.Tensor,
        selected_batches: List[torch.Tensor],
        max_t: int
    ) -> torch.Tensor:
        """
        Dynamically computes transformer attention modulation weights (B, max_t)
        based on spatial movement velocity and temporal entropy.
        """
        b = len(selected_batches)
        if self.uniform_sampling:
            return torch.ones((b, max_t), device=scores.device, dtype=scores.dtype)
        mod_weights = torch.zeros((b, max_t), device=scores.device, dtype=scores.dtype)
        for i in range(b):
            chosen = selected_batches[i]
            s = scores[i, chosen]
            if s.numel() > 1 and s.std() > 1e-6:
                s_norm = (s - s.mean()) / (s.std() + 1e-6)
            else:
                s_norm = torch.zeros_like(s)
            m = torch.sigmoid(s_norm)
            k = m.shape[0]
            if k < max_t:
                m = torch.cat([m, m[-1:].expand(max_t - k)])
            elif k > max_t:
                m = m[:max_t]
            mod_weights[i] = m
        return mod_weights
