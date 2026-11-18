"""
Dynamic Scene Graph Generation & Relational Edge-Conditioned Graph Attention (EdgeGAT).

Implements Modules 3 & 4 of the ARGUS research design:
  - Bounding box detection and dynamic interaction scene graph construction.
  - Models Person-to-Person and Person-to-Object relationships explicitly in edge attributes.
  - Vectorized Multi-Head Edge-Conditioned Graph Attention Network (EdgeGAT).
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import (
    NUM_SEMANTIC_CLASSES,
    SEMANTIC_TYPE_PERSON,
    SEMANTIC_TYPE_VEHICLE,
    SEMANTIC_TYPE_BAG,
    SEMANTIC_TYPE_WEAPON,
    SEMANTIC_TYPE_OTHER,
)

# ── Shared Graph Dimension Constants ───────────────────────────────────
KINEMATIC_NODE_DIM = 6                                              # centers (2) + sizes (2) + velocity (2)
SEMANTIC_NODE_DIM = KINEMATIC_NODE_DIM + NUM_SEMANTIC_CLASSES + 1  # 6 + 5 + 1 = 12
LEGACY_SEMANTIC_NODE_DIM = 10                                       # 6 + 3 + 1 (legacy3 schema)

KINEMATIC_EDGE_DIM = 4                                              # dist, iou, rel_vel, persistence
RELATIONAL_EDGE_DIM = 7                                             # 4 kinematic + 3 broad (P2P, P2O, O2O)
THREAT_RELATIONAL_EDGE_DIM = 10                                     # 4 kinematic + 3 broad + 3 threat (P2W, P2B, W2O)


class GridDetector:
    """Fast non-learned heuristic detector using spatial gradient and variance."""
    def __init__(self, max_objects: int = 12):
        self.max_objects = max_objects

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        h, w = frame.shape[-2:]
        gray = frame.mean(0)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, :-1] = torch.abs(gray[:, 1:] - gray[:, :-1])
        gy[:-1, :] = torch.abs(gray[1:, :] - gray[:-1, :])
        energy = gx + gy
        cell = max(8, min(h, w) // 6)
        if h < cell or w < cell:
            return torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=frame.device, dtype=torch.float32)

        e_blocks = energy.unfold(0, cell, cell).unfold(1, cell, cell)
        g_blocks = gray.unfold(0, cell, cell).unfold(1, cell, cell)
        scores = (e_blocks.sum(dim=(-2, -1)) + g_blocks.var(dim=(-2, -1))).reshape(-1)
        nx = g_blocks.shape[1]
        k = min(self.max_objects, scores.numel())
        _, top_idx = torch.topk(scores, k)
        iy = (top_idx // nx).float()
        ix = (top_idx % nx).float()
        x1 = (ix * cell) / w
        y1 = (iy * cell) / h
        x2 = (ix * cell + cell) / w
        y2 = (iy * cell + cell) / h
        return torch.stack([x1, y1, x2, y2], dim=-1).to(dtype=torch.float32)


class DynamicSceneGraphBuilder:
    """
    Constructs an interaction scene graph for a video frame.
    Supports Person-to-Person (P2P), Person-to-Object (P2O), and Object-to-Object (O2O)
    relational conditioning.
    """
    def __init__(
        self,
        max_objects: int = 12,
        dist_threshold: float = 0.6,
        detector=None,
        use_semantic: bool = False,
        tracking_iou_threshold: float = 0.1,
        semantic_schema_version: str = "threat5",
    ):
        self.max_objects = max_objects
        self.dist_threshold = dist_threshold
        self.use_semantic = use_semantic
        self.tracking_iou_threshold = tracking_iou_threshold
        self.semantic_schema_version = semantic_schema_version
        if detector is not None:
            self.detector = detector
        elif use_semantic:
            from .detector import SemanticDetector
            self.detector = SemanticDetector(max_objects=max_objects)
        else:
            self.detector = GridDetector(max_objects)

    def build(
        self,
        frame: torch.Tensor,
        prev_boxes: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        detection_result=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Build scene graph for a single frame.

        Args:
            frame: (3, H, W) normalized image.
            prev_boxes: (N_prev, 4) boxes from prior frame for velocity estimation.
            visual_features: (N, D_vis) optional RoI visual features.
            detection_result: Pre-computed detection result to avoid re-running detector.
        Returns:
            node_feats: (N, D_node)
            edges: (src, dst) long tensors
            edge_attr: (E, D_edge)
            boxes: (N, 4) bounding boxes
        """
        # Detection step — use pre-computed result if available
        sem_types = None
        scores = None
        if detection_result is not None:
            if hasattr(detection_result, 'boxes'):
                boxes = detection_result.boxes
                sem_types = detection_result.semantic_types
                scores = detection_result.scores
            else:
                boxes = detection_result
        elif hasattr(self.detector, 'forward') and callable(getattr(self.detector, 'forward', None)):
            det_result = self.detector(frame)
            if hasattr(det_result, 'boxes'):
                boxes = det_result.boxes
                sem_types = det_result.semantic_types
                scores = det_result.scores
            else:
                boxes = det_result
        else:
            boxes = self.detector(frame)

        n = len(boxes)
        visual_dim = visual_features.shape[1] if visual_features is not None and visual_features.ndim == 2 else 0

        # Ensure semantic types exist if semantic graph is enabled
        if self.use_semantic and sem_types is None and n > 0:
            sem_types = torch.full((n,), SEMANTIC_TYPE_OTHER, dtype=torch.long, device=frame.device)
            if scores is None:
                scores = torch.ones((n,), device=frame.device)

        if sem_types is not None:
            if not isinstance(sem_types, torch.Tensor):
                sem_types = torch.as_tensor(sem_types, device=frame.device, dtype=torch.long)
            sem_types = sem_types.view(-1)

        if scores is not None:
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, device=frame.device, dtype=torch.float32)
            scores = scores.view(-1)

        # Calculate exact node and edge dimensions based on schema
        if self.use_semantic:
            base_node_dim = SEMANTIC_NODE_DIM if self.semantic_schema_version == "threat5" else LEGACY_SEMANTIC_NODE_DIM
            edge_dim = THREAT_RELATIONAL_EDGE_DIM if self.semantic_schema_version == "threat5" else RELATIONAL_EDGE_DIM
        else:
            base_node_dim = KINEMATIC_NODE_DIM
            edge_dim = KINEMATIC_EDGE_DIM
        node_dim = base_node_dim + visual_dim

        if n == 0:
            empty_nodes = torch.empty((0, node_dim), device=frame.device, dtype=frame.dtype)
            empty_idx = torch.empty((0,), device=frame.device, dtype=torch.long)
            empty_edges = torch.empty((0, edge_dim), device=frame.device, dtype=frame.dtype)
            return empty_nodes, (empty_idx, empty_idx), empty_edges, boxes
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)

        if prev_boxes is not None and len(prev_boxes) > 0:
            prev_centers = (prev_boxes[:, :2] + prev_boxes[:, 2:]) / 2.0
            velocity = torch.zeros_like(centers)
            # Greedy IoU association prevents detector ordering changes from being
            # interpreted as object motion.
            ix1 = torch.maximum(boxes[:, 0:1], prev_boxes[:, 0:1].T)
            iy1 = torch.maximum(boxes[:, 1:2], prev_boxes[:, 1:2].T)
            ix2 = torch.minimum(boxes[:, 2:3], prev_boxes[:, 2:3].T)
            iy2 = torch.minimum(boxes[:, 3:4], prev_boxes[:, 3:4].T)
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            area_current = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            area_previous = ((prev_boxes[:, 2] - prev_boxes[:, 0]).clamp(min=0)
                             * (prev_boxes[:, 3] - prev_boxes[:, 1]).clamp(min=0))
            iou_mat = (inter / (area_current[:, None] + area_previous[None, :] - inter + 1e-6)).tolist()
            used_prev: set[int] = set()
            for current_idx in range(n):
                row_ious = iou_mat[current_idx]
                best_iou = -1.0
                best_idx = -1
                for prev_idx, val in enumerate(row_ious):
                    if prev_idx not in used_prev and val > best_iou:
                        best_iou = val
                        best_idx = prev_idx
                if best_idx >= 0 and best_iou >= self.tracking_iou_threshold:
                    used_prev.add(best_idx)
                    velocity[current_idx] = centers[current_idx] - prev_centers[best_idx]
        else:
            velocity = torch.zeros_like(centers)

        # Build node features
        if self.use_semantic and sem_types is not None:
            num_classes = NUM_SEMANTIC_CLASSES if self.semantic_schema_version == "threat5" else 3
            sem_one_hot = F.one_hot(sem_types.clamp(0, num_classes - 1), num_classes=num_classes).float().to(frame.device)
            conf = scores.unsqueeze(-1).to(frame.device) if scores is not None else torch.ones((n, 1), device=frame.device)
            node_parts = [centers, sizes, velocity, sem_one_hot, conf]
            if visual_features is not None and visual_features.shape[0] == n:
                node_parts.append(visual_features)
            node_feats = torch.cat(node_parts, dim=-1)
        else:
            # Kinematic 6-dim node features: [centers, sizes, velocity]
            node_parts = [centers, sizes, velocity]
            if visual_features is not None and visual_features.shape[0] == n:
                node_parts.append(visual_features)
            node_feats = torch.cat(node_parts, dim=-1)

        # Build edges and relational attributes (vectorized on device)
        dist_mat = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=-1)
        eye = torch.eye(n, device=frame.device, dtype=torch.bool)
        adj = (dist_mat <= self.dist_threshold) & (~eye)
        src_t, dst_t = adj.nonzero(as_tuple=True)
        if src_t.numel() == 0:
            src_t = torch.arange(n, device=frame.device, dtype=torch.long)
            dst_t = src_t

        d_t = dist_mat[src_t, dst_t].float()
        bi = boxes[src_t]
        bj = boxes[dst_t]
        ix1 = torch.maximum(bi[:, 0], bj[:, 0])
        iy1 = torch.maximum(bi[:, 1], bj[:, 1])
        ix2 = torch.minimum(bi[:, 2], bj[:, 2])
        iy2 = torch.minimum(bi[:, 3], bj[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_i = (bi[:, 2] - bi[:, 0]) * (bi[:, 3] - bi[:, 1])
        area_j = (bj[:, 2] - bj[:, 0]) * (bj[:, 3] - bj[:, 1])
        iou_t = (inter / (area_i + area_j - inter + 1e-6)).float()
        rel_vel_t = torch.norm(velocity[src_t] - velocity[dst_t], dim=-1).float()
        persist_t = torch.ones_like(d_t)

        if self.use_semantic and sem_types is not None:
            ti = sem_types[src_t].to(frame.device)
            tj = sem_types[dst_t].to(frame.device)
            ti_p = ti == SEMANTIC_TYPE_PERSON
            tj_p = tj == SEMANTIC_TYPE_PERSON
            is_p2p = (ti_p & tj_p).float()
            is_p2o = (ti_p ^ tj_p).float()
            is_o2o = ((~ti_p) & (~tj_p)).float()

            if self.semantic_schema_version == "threat5":
                ti_w = ti == SEMANTIC_TYPE_WEAPON
                tj_w = tj == SEMANTIC_TYPE_WEAPON
                ti_b = ti == SEMANTIC_TYPE_BAG
                tj_b = tj == SEMANTIC_TYPE_BAG
                is_p2w = ((ti_p & tj_w) | (ti_w & tj_p)).float()
                is_p2b = ((ti_p & tj_b) | (ti_b & tj_p)).float()
                is_w2o = ((ti_w & (~tj_p)) | (tj_w & (~ti_p))).float()
                edge_attr = torch.stack(
                    [d_t, iou_t, rel_vel_t, persist_t, is_p2p, is_p2o, is_o2o, is_p2w, is_p2b, is_w2o],
                    dim=-1,
                )
            else:
                edge_attr = torch.stack(
                    [d_t, iou_t, rel_vel_t, persist_t, is_p2p, is_p2o, is_o2o],
                    dim=-1,
                )
        else:
            edge_attr = torch.stack([d_t, iou_t, rel_vel_t, persist_t], dim=-1)

        return node_feats, (src_t, dst_t), edge_attr, boxes


class EdgeGAT(nn.Module):
    """
    Edge-Conditioned Multi-Head Graph Attention Network.
    Fully vectorized with PyTorch scatter operations for fast training and inference.
    Supports standard GAT mode without edge attributes (Ablation 2).
    """
    def __init__(self, in_dim: int = 6, edge_dim: int = 4, out_dim: int = 128, heads: int = 4, use_edge_attrs: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.use_edge_attrs = use_edge_attrs
        self.proj = nn.Linear(in_dim, out_dim)
        if self.use_edge_attrs:
            self.edge_proj = nn.Linear(edge_dim, out_dim)
            self.attn = nn.Parameter(torch.randn(heads, self.head_dim * 3))
        else:
            self.edge_proj = None
            self.attn = nn.Parameter(torch.randn(heads, self.head_dim * 2))
        nn.init.xavier_uniform_(self.attn)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor], edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 0:
            return torch.zeros((0, self.heads * self.head_dim), device=x.device, dtype=x.dtype)
        h = self.proj(x).view(n, self.heads, self.head_dim)
        src, dst = edges
        if src.numel() == 0:
            return self.out_proj(h.view(n, -1))

        h_src = h[src]
        h_dst = h[dst]

        if self.use_edge_attrs and self.edge_proj is not None and edge_attr is not None and edge_attr.numel() > 0:
            e_feat = self.edge_proj(edge_attr).view(-1, self.heads, self.head_dim)
            combined = torch.cat([h_src, h_dst, e_feat], dim=-1)
        else:
            e_feat = None
            combined = torch.cat([h_src, h_dst], dim=-1)

        logits = (combined * self.attn.unsqueeze(0)).sum(dim=-1)
        logits = F.leaky_relu(logits, 0.2)

        # Vectorized scatter-based attention
        num_edges = logits.shape[0]
        if num_edges == 0:
            return self.out_proj(h.view(n, -1))

        # Scatter-max for numerical stability
        max_logits = logits.new_full((n, self.heads), float('-inf'))
        max_logits.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(logits), logits, reduce='amax', include_self=True)
        max_logits = max_logits.clamp(min=-1e9)

        exp_logits = torch.exp(logits - max_logits[dst])

        sum_exp = torch.zeros(n, self.heads, device=x.device, dtype=x.dtype)
        sum_exp.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_logits), exp_logits)
        sum_exp = sum_exp.clamp(min=1e-8)

        alpha = (exp_logits / sum_exp[dst]).unsqueeze(-1)

        if e_feat is not None:
            msg = (h_src + e_feat) * alpha
        else:
            msg = h_src * alpha
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)

        # Self-loop for isolated nodes (no incoming edges)
        incoming_count = torch.zeros(n, device=x.device)
        incoming_count.scatter_add_(0, dst, torch.ones(num_edges, device=x.device))
        isolated = (incoming_count == 0)
        if isolated.any():
            out[isolated] = h[isolated]

        out = out.view(n, -1)
        return self.out_proj(out)
