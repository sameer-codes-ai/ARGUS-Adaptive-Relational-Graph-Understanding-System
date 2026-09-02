import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphVisualFusion(nn.Module):
    def __init__(self, visual_dim=192, graph_dim=128):
        super().__init__()
        self.g_proj = nn.Linear(graph_dim, visual_dim)
        self.attn = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(visual_dim * 2, visual_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_tokens, graph_tokens):
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
    def __init__(self, visual_dim=192, keep_ratio=0.5):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Linear(visual_dim // 2, 1)
        )
        self.keep_ratio = keep_ratio

    def forward(self, tokens, graph_affinity=None):
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
    def __init__(self, dim=192, size=16):
        super().__init__()
        self.dim = dim
        self.size = size
        self.register_buffer('bank', None)

    def reset(self):
        self.bank = None

    def forward(self, latest, update=True):
        if self.bank is None or self.bank.shape[0] != latest.shape[0]:
            enriched = latest
        else:
            enriched = torch.cat([self.bank.to(latest.device), latest], dim=1)

        if update:
            with torch.no_grad():
                if self.bank is None or self.bank.shape[0] != latest.shape[0]:
                    new_bank = latest.detach()
                else:
                    new_bank = torch.cat([self.bank.to(latest.device), latest.detach()], dim=1)
                self.bank = new_bank[:, -self.size:]
        return enriched


class TemporalEncoder(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x):
        t = x.shape[1]
        mask = torch.full((t, t), float('-inf'), device=x.device)
        mask = torch.triu(mask, 1)
        return self.encoder(x, mask=mask)
