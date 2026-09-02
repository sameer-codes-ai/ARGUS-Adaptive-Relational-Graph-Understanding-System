from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
from .config import ARGUSConfig
from .sampling import AdaptiveSampler
from .graph import DynamicSceneGraphBuilder, EdgeGAT
from .modules import GraphVisualFusion, GraphGuidedPruner, TemporalMemory, TemporalEncoder

@dataclass
class ARGUSOutput:
    scores: torch.Tensor
    sampled_indices: List[torch.Tensor]
    boxes: List[List[torch.Tensor]]
    saliency: Optional[List[torch.Tensor]] = None
    node_importance: Optional[List[torch.Tensor]] = None

class ARGUS(nn.Module):
    def __init__(self, config: Optional[ARGUSConfig] = None):
        super().__init__()
        self.cfg = config or ARGUSConfig()
        self.sampler = AdaptiveSampler(
            threshold=self.cfg.sampling_threshold,
            max_gap=self.cfg.max_frame_gap
        )
        self.graph_builder = DynamicSceneGraphBuilder(max_objects=self.cfg.max_objects)
        self.edge_gat = EdgeGAT(in_dim=6, edge_dim=4, out_dim=self.cfg.graph_dim, heads=self.cfg.heads)

        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.cfg.dim,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size
        )
        self.fusion = GraphVisualFusion(visual_dim=self.cfg.dim, graph_dim=self.cfg.graph_dim)
        self.pruner = GraphGuidedPruner(visual_dim=self.cfg.dim, keep_ratio=self.cfg.keep_ratio)
        self.temporal_encoder = TemporalEncoder(
            dim=self.cfg.dim,
            layers=self.cfg.temporal_layers,
            heads=self.cfg.heads
        )
        self.memory = TemporalMemory(dim=self.cfg.dim, size=self.cfg.memory_size)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim // 2),
            nn.GELU(),
            nn.Linear(self.cfg.dim // 2, 1),
            nn.Sigmoid()
        )

    def _encode_frame(self, frame, prev_boxes=None):
        nodes, edges, edge_attr, boxes = self.graph_builder.build(frame, prev_boxes)
        g_embed = self.edge_gat(nodes, edges, edge_attr)

        patches = self.patch_proj(frame.unsqueeze(0))
        patches = patches.flatten(2).transpose(1, 2)

        fused = self.fusion(patches, g_embed)
        pruned, topk_scores, topk_idx = self.pruner(fused)
        frame_token = pruned.mean(dim=1)
        return frame_token, boxes, topk_scores, g_embed

    def forward(self, frames, stream: bool = False):
        b, t = frames.shape[:2]
        selected_batches, raw_scores = self.sampler.select(frames)

        batch_tokens = []
        batch_boxes = []
        batch_saliencies = []
        batch_node_imp = []

        for i in range(b):
            chosen = selected_batches[i]
            prev_boxes = None
            clip_tokens = []
            clip_boxes = []
            clip_saliencies = []
            clip_node_imp = []

            for frame_idx in chosen:
                f_token, boxes, saliency, node_emb = self._encode_frame(frames[i, frame_idx], prev_boxes)
                prev_boxes = boxes.detach()
                clip_tokens.append(f_token)
                clip_boxes.append(boxes)
                clip_saliencies.append(saliency)
                clip_node_imp.append(node_emb)

            clip_tokens_t = torch.cat(clip_tokens, dim=0).unsqueeze(0)
            batch_tokens.append(clip_tokens_t)
            batch_boxes.append(clip_boxes)
            batch_saliencies.append(clip_saliencies)
            batch_node_imp.append(clip_node_imp)

        max_t = max(bt.shape[1] for bt in batch_tokens)
        padded_tokens = []
        for bt in batch_tokens:
            if bt.shape[1] < max_t:
                pad = bt[:, -1:].expand(-1, max_t - bt.shape[1], -1)
                padded = torch.cat([bt, pad], dim=1)
            else:
                padded = bt
            padded_tokens.append(padded)
        batch_tokens_t = torch.cat(padded_tokens, dim=0)

        enriched = self.memory(batch_tokens_t, update=stream)
        encoded = self.temporal_encoder(enriched)
        current_encoded = encoded[:, -max_t:]
        pred_scores = self.head(current_encoded).squeeze(-1)

        return ARGUSOutput(
            scores=pred_scores,
            sampled_indices=selected_batches,
            boxes=batch_boxes,
            saliency=batch_saliencies,
            node_importance=batch_node_imp
        )
