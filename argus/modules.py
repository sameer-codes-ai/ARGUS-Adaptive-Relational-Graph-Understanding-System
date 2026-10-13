"""
Neural Building Blocks for ARGUS.

Implements Modules 4, 5, 6, 7 of the ARGUS research design:
  - GraphVisualFusion: Cross-attention and gated residual between visual and graph tokens.
  - GraphGuidedPruner: Topology-guided token scoring and retention.
  - TemporalMemory: Bounded FIFO streaming buffer.
  - TemporalEncoder: Causal Multi-Head Transformer Encoder with Dynamic Attention Modulation.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphVisualFusion(nn.Module):
    """
    Fuses dense visual patch tokens with structural scene graph embeddings
    via gated cross-attention.
    """
    def __init__(self, visual_dim: int = 192, graph_dim: int = 128):
        super().__init__()
        self.g_proj = nn.Linear(graph_dim, visual_dim)
        self.attn = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(visual_dim * 2, visual_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_tokens: torch.Tensor, graph_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        if graph_tokens.ndim == 2:
            graph_tokens = graph_tokens.unsqueeze(0)

        g_proj = self.g_proj(graph_tokens)
        attended, _ = self.attn(visual_tokens, g_proj, g_proj)
        gate_val = self.gate(torch.cat([visual_tokens, attended], dim=-1))
        fused = self.norm(visual_tokens + gate_val * attended)
        return fused


class GraphGuidedPruner(nn.Module):
    """
    Prunes uninformative visual background tokens guided by graph affinity.
    """
    def __init__(self, visual_dim: int = 192, keep_ratio: float = 0.5):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Linear(visual_dim // 2, 1)
        )
        self.keep_ratio = keep_ratio

    def forward(
        self,
        tokens: torch.Tensor,
        graph_affinity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.score_net(tokens).squeeze(-1)
        if graph_affinity is not None:
            scores = scores + 0.3 * graph_affinity

        p = tokens.shape[1]
        k = max(1, int(p * self.keep_ratio))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        pruned_tokens = torch.gather(tokens, dim=1, index=topk_indices_exp)
        return pruned_tokens, topk_scores, topk_indices


class TemporalMemory(nn.Module):
    """
    Bounded streaming FIFO memory bank for causal video evaluation.
    """
    def __init__(self, dim: int = 192, size: int = 16):
        super().__init__()
        self.dim = dim
        self.size = size
        self.register_buffer('bank', None)

    def reset(self):
        self.bank = None

    def forward(self, latest: torch.Tensor, update: bool = True) -> torch.Tensor:
        b = latest.shape[0]
        if self.bank is not None and self.bank.shape[0] != b:
            if b < self.bank.shape[0]:
                self.bank = self.bank[:b]
            else:
                self.bank = None

        if self.bank is None:
            enriched = latest
        else:
            enriched = torch.cat([self.bank.to(latest.device), latest], dim=1)

        if update:
            with torch.no_grad():
                if self.bank is None:
                    new_bank = latest.detach()
                else:
                    new_bank = torch.cat([self.bank.to(latest.device), latest.detach()], dim=1)
                self.bank = new_bank[:, -self.size:]
        return enriched


class TemporalEncoder(nn.Module):
    """
    Causal Temporal Transformer with Dynamic Attention Modulation.
    Modulates attention logits based on spatial movement velocity and temporal entropy.
    """
    def __init__(self, dim: int, layers: int, heads: int):
        super().__init__()
        self.heads = heads
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = x.shape[:2]
        # Base causal upper-triangular mask
        mask = torch.triu(torch.full((t, t), float('-inf'), device=x.device), 1)

        if modulation is not None:
            # Dynamic attention modulation from spatial velocity and temporal entropy
            if modulation.shape[1] < t:
                pad = torch.full((b, t - modulation.shape[1]), 0.5, device=x.device, dtype=x.dtype)
                mod_full = torch.cat([pad, modulation.to(x.dtype)], dim=1)
            else:
                mod_full = modulation[:, -t:].to(x.dtype)
            # Modulate token scale by dynamism factor
            scale = 1.0 + 0.5 * (mod_full.unsqueeze(-1) - 0.5)
            x = x * scale

        return self.encoder(x, mask=mask)
