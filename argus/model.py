"""
ARGUS: Adaptive Relational Graph Understanding System for Surveillance.

End-to-end architecture unifying:
  - Adaptive Frame-Sampling Controller with Attention Modulation (Module 1)
  - Pre-trained / Native ConvNeXt Spatial Feature Backbone (Module 2)
  - Semantic Scene Graph Builder with Person-to-Person / Person-to-Object Relations (Module 3)
  - Relational Edge-Conditioned Graph Attention (EdgeGAT) (Module 4)
  - Topology-Aware Graph-Visual Fusion & Pruning (Modules 5 & 6)
  - Causal Temporal Transformer with Streaming FIFO Memory (Module 7)
  - Real-Time Anomaly Scoring Engine (Module 8)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARGUSConfig
from .sampling import AdaptiveSampler
from .backbone import SpatialBackbone
from .detector import SemanticDetector, RoIFeatureExtractor
from .graph import DynamicSceneGraphBuilder, EdgeGAT
from .modules import GraphVisualFusion, GraphGuidedPruner, TemporalMemory, TemporalEncoder


@dataclass
class ARGUSOutput:
    scores: torch.Tensor
    sampled_indices: List[torch.Tensor]
    boxes: List[List[torch.Tensor]]
    saliency: Optional[List[List[torch.Tensor]]] = None
    node_importance: Optional[List[List[torch.Tensor]]] = None
    valid_mask: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None


class ARGUS(nn.Module):
    def __init__(self, config: Optional[ARGUSConfig] = None):
        super().__init__()
        self.cfg = (config or ARGUSConfig()).validate()

        # ── Module 1: Adaptive Frame Sampling Controller ─────────────────
        self.sampler = AdaptiveSampler(
            threshold=self.cfg.sampling_threshold,
            max_gap=self.cfg.max_frame_gap
        )

        # ── Module 2: Spatial Feature Backbone ───────────────────────────
        if self.cfg.use_pretrained_backbone:
            self.backbone = SpatialBackbone(
                name=self.cfg.backbone_name,
                out_dim=self.cfg.dim,
                frozen=self.cfg.backbone_frozen,
                pretrained=self.cfg.backbone_pretrained,
                allow_fallback=self.cfg.allow_component_fallbacks,
            )
            self.patch_proj = None
        else:
            self.backbone = None
            self.patch_proj = nn.Conv2d(
                in_channels=3,
                out_channels=self.cfg.dim,
                kernel_size=self.cfg.patch_size,
                stride=self.cfg.patch_size
            )

        # ── Module 3: Semantic Scene Graph Construction ─────────────────
        if self.cfg.use_semantic_detector:
            self.detector = SemanticDetector(
                confidence=self.cfg.detector_confidence,
                max_objects=self.cfg.max_objects,
                model_name=self.cfg.detector_name,
                allow_fallback=self.cfg.allow_component_fallbacks,
            )
            in_node_dim = 10  # 6 kinematic + 3 semantic one-hot + 1 conf
            edge_attr_dim = 7  # 4 kinematic + 3 relational (P2P, P2O, O2O)
            self.graph_builder = DynamicSceneGraphBuilder(
                max_objects=self.cfg.max_objects,
                detector=self.detector,
                use_semantic=True,
                tracking_iou_threshold=self.cfg.tracking_iou_threshold,
            )
        else:
            self.detector = None
            in_node_dim = 6   # 6 kinematic (center, size, velocity)
            edge_attr_dim = 4  # 4 kinematic (dist, iou, rel_vel, persist)
            self.graph_builder = DynamicSceneGraphBuilder(
                max_objects=self.cfg.max_objects,
                use_semantic=False,
                tracking_iou_threshold=self.cfg.tracking_iou_threshold,
            )

        # Optional RoI visual feature extractor for graph nodes
        if self.cfg.node_visual_dim > 0:
            feat_chans = self.backbone.feat_dim if self.backbone is not None else self.cfg.dim
            self.roi_extractor = RoIFeatureExtractor(
                in_channels=feat_chans,
                out_dim=self.cfg.node_visual_dim
            )
            in_node_dim += self.cfg.node_visual_dim
        else:
            self.roi_extractor = None

        # ── Module 4: Relational Edge-Conditioned Graph Attention ─────────
        self.edge_gat = EdgeGAT(
            in_dim=in_node_dim,
            edge_dim=edge_attr_dim,
            out_dim=self.cfg.graph_dim,
            heads=self.cfg.heads
        )

        # ── Module 5: Graph-Visual Cross-Attention Fusion ────────────────
        self.fusion = GraphVisualFusion(
            visual_dim=self.cfg.dim,
            graph_dim=self.cfg.graph_dim,
            heads=self.cfg.heads,
        )

        # ── Module 6: Graph-Guided Visual Token Pruning ──────────────────
        self.pruner = GraphGuidedPruner(
            visual_dim=self.cfg.dim,
            keep_ratio=self.cfg.keep_ratio
        )

        # ── Module 7: Causal Temporal Transformer with Modulation ────────
        self.temporal_encoder = TemporalEncoder(
            dim=self.cfg.dim,
            layers=self.cfg.temporal_layers,
            heads=self.cfg.heads
        )
        self.memory = TemporalMemory(dim=self.cfg.dim, size=self.cfg.memory_size)

        # ── Module 8: Anomaly Scoring Head ──────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim // 2),
            nn.GELU(),
            nn.Linear(self.cfg.dim // 2, 1),
            nn.Sigmoid()
        )

    def component_metadata(self) -> dict:
        backbone_impl = "patch_projection"
        if self.backbone is not None:
            backbone_impl = "timm" if self.backbone.use_timm else "native_convnext"
        detector_impl = "grid"
        if self.detector is not None:
            detector_impl = "yolo" if self.detector.has_yolo else "heuristic"
        return {
            "backbone_implementation": backbone_impl,
            "backbone_name": self.cfg.backbone_name if self.backbone is not None else None,
            "backbone_pretrained_requested": self.cfg.backbone_pretrained,
            "detector_implementation": detector_impl,
            "detector_model": self.detector.model_name if self.detector is not None else None,
        }

    def _encode_frame(self, frame: torch.Tensor, prev_boxes: Optional[torch.Tensor] = None):
        # 1. Visual Token Extraction
        if self.backbone is not None:
            patches, feat_maps = self.backbone(frame)
            active_feat_map = feat_maps[self.backbone.token_stage_idx]
        else:
            p = self.patch_proj(frame.unsqueeze(0))
            patches = p.flatten(2).transpose(1, 2)
            active_feat_map = p

        # 2. Run detection ONCE (shared between RoI and graph builder)
        det_result = self.graph_builder.detector(frame)
        if hasattr(det_result, 'boxes'):
            det_boxes = det_result.boxes
        else:
            det_boxes = det_result

        # 3. RoI features (if enabled)
        vis_features = None
        if self.roi_extractor is not None:
            vis_features = self.roi_extractor(active_feat_map, det_boxes)

        # 4. Scene Graph Construction & Attention (pass pre-computed detection)
        nodes, edges, edge_attr, boxes = self.graph_builder.build(
            frame, prev_boxes=prev_boxes, visual_features=vis_features,
            detection_result=det_result
        )
        if nodes.shape[0] == 0:
            g_embed = torch.zeros((1, self.cfg.graph_dim), device=frame.device, dtype=patches.dtype)
        else:
            g_embed = self.edge_gat(nodes, edges, edge_attr)

        # 4. Graph-Visual Fusion
        fused = self.fusion(patches, g_embed)

        # 5. Graph-Guided Pruning
        g_proj = self.fusion.g_proj(g_embed)
        if g_proj.ndim == 2:
            g_proj = g_proj.unsqueeze(0)
        fused_norm = F.normalize(fused, dim=-1)
        g_norm = F.normalize(g_proj, dim=-1)
        sim = torch.bmm(fused_norm, g_norm.transpose(1, 2))
        graph_affinity = sim.max(dim=-1).values

        pruned, topk_scores, topk_idx = self.pruner(fused, graph_affinity=graph_affinity)
        frame_token = pruned.mean(dim=1)
        return frame_token, boxes, topk_scores, g_embed

    def forward(
        self,
        frames: torch.Tensor,
        stream: bool = False,
        stream_id: str = "default",
    ) -> ARGUSOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape (B, T, 3, H, W)")
        b, t = frames.shape[:2]
        if b == 0 or t == 0:
            raise ValueError("frames must contain at least one batch item and one frame")
        if stream and b != 1:
            raise ValueError("streaming inference requires batch size 1")
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
        lengths = torch.tensor([bt.shape[1] for bt in batch_tokens], device=frames.device, dtype=torch.long)
        padded_tokens = []
        for bt in batch_tokens:
            if bt.shape[1] < max_t:
                pad = bt[:, -1:].expand(-1, max_t - bt.shape[1], -1)
                padded = torch.cat([bt, pad], dim=1)
            else:
                padded = bt
            padded_tokens.append(padded)
        batch_tokens_t = torch.cat(padded_tokens, dim=0)
        valid_mask = torch.arange(max_t, device=frames.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Dynamic Attention Modulation from spatial velocity & temporal entropy
        mod_weights = self.sampler.compute_attention_modulation(raw_scores, selected_batches, max_t)

        if stream:
            enriched = self.memory(batch_tokens_t, update=True, stream_id=stream_id)
            encoded = self.temporal_encoder(enriched, modulation=mod_weights)
        else:
            enriched = batch_tokens_t
            encoded = self.temporal_encoder(
                enriched,
                modulation=mod_weights,
                padding_mask=~valid_mask,
            )
        current_encoded = encoded[:, -max_t:]
        pred_scores = self.head(current_encoded).squeeze(-1)
        pred_scores = pred_scores.masked_fill(~valid_mask, 0.0)

        return ARGUSOutput(
            scores=pred_scores,
            sampled_indices=selected_batches,
            boxes=batch_boxes,
            saliency=batch_saliencies,
            node_importance=batch_node_imp,
            valid_mask=valid_mask,
            sequence_lengths=lengths,
        )
