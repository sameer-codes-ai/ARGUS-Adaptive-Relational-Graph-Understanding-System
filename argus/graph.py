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
        candidates = []
        for y in range(0, h - cell + 1, cell):
            for x in range(0, w - cell + 1, cell):
                score = energy[y:y + cell, x:x + cell].sum().item() + gray[y:y + cell, x:x + cell].var().item()
                candidates.append((score, x, y, x + cell, y + cell))
        candidates.sort(key=lambda a: float(a[0]), reverse=True)
        boxes = torch.tensor(
            [[x / w, y / h, x2 / w, y2 / h] for _, x, y, x2, y2 in candidates[:self.max_objects]],
            device=frame.device,
            dtype=torch.float32
        )
        if len(boxes) == 0:
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=frame.device, dtype=torch.float32)
        return boxes


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
            used_prev: set[int] = set()
            for current_idx in range(n):
                box = boxes[current_idx]
                ix1 = torch.maximum(box[0], prev_boxes[:, 0])
                iy1 = torch.maximum(box[1], prev_boxes[:, 1])
                ix2 = torch.minimum(box[2], prev_boxes[:, 2])
                iy2 = torch.minimum(box[3], prev_boxes[:, 3])
                inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                area_current = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
                area_previous = ((prev_boxes[:, 2] - prev_boxes[:, 0]).clamp(min=0)
                                 * (prev_boxes[:, 3] - prev_boxes[:, 1]).clamp(min=0))
                ious = inter / (area_current + area_previous - inter + 1e-6)
                for prev_idx in used_prev:
                    ious[prev_idx] = -1.0
                best_iou, best_idx = ious.max(dim=0)
                if float(best_iou) >= self.tracking_iou_threshold:
                    matched = int(best_idx)
                    used_prev.add(matched)
                    velocity[current_idx] = centers[current_idx] - prev_centers[matched]
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

        # Build edges and relational attributes
        edges_src, edges_dst, edge_attrs = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = torch.norm(centers[i] - centers[j])
                if d <= self.dist_threshold:
                    bi = boxes[i]
                    bj = boxes[j]
                    ix1, iy1 = torch.max(bi[0], bj[0]), torch.max(bi[1], bj[1])
                    ix2, iy2 = torch.min(bi[2], bj[2]), torch.min(bi[3], bj[3])
                    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                    area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                    area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
                    union = area_i + area_j - inter + 1e-6
                    iou = inter / union
                    rel_vel = torch.norm(velocity[i] - velocity[j])

                    if self.use_semantic and sem_types is not None:
                        ti = int(sem_types[i].item() if hasattr(sem_types[i], 'item') else sem_types[i])
                        tj = int(sem_types[j].item() if hasattr(sem_types[j], 'item') else sem_types[j])
                        is_p2p = 1.0 if (ti == SEMANTIC_TYPE_PERSON and tj == SEMANTIC_TYPE_PERSON) else 0.0
                        is_p2o = 1.0 if ((ti == SEMANTIC_TYPE_PERSON and tj != SEMANTIC_TYPE_PERSON) or (ti != SEMANTIC_TYPE_PERSON and tj == SEMANTIC_TYPE_PERSON)) else 0.0
                        is_o2o = 1.0 if (ti != SEMANTIC_TYPE_PERSON and tj != SEMANTIC_TYPE_PERSON) else 0.0

                        if self.semantic_schema_version == "threat5":
                            # Threat-aware relations:
                            # Person-to-Weapon (P2W)
                            is_p2w = 1.0 if ((ti == SEMANTIC_TYPE_PERSON and tj == SEMANTIC_TYPE_WEAPON) or (ti == SEMANTIC_TYPE_WEAPON and tj == SEMANTIC_TYPE_PERSON)) else 0.0
                            # Person-to-Bag (P2B)
                            is_p2b = 1.0 if ((ti == SEMANTIC_TYPE_PERSON and tj == SEMANTIC_TYPE_BAG) or (ti == SEMANTIC_TYPE_BAG and tj == SEMANTIC_TYPE_PERSON)) else 0.0
                            # Weapon-to-Object (W2O): weapon interacting with any non-person entity
                            is_w2o = 1.0 if ((ti == SEMANTIC_TYPE_WEAPON and tj != SEMANTIC_TYPE_PERSON) or (tj == SEMANTIC_TYPE_WEAPON and ti != SEMANTIC_TYPE_PERSON)) else 0.0
                            # 4 kinematic + 3 broad relational + 3 threat-aware relational = 10 dims
                            edge_attrs.append([d.item(), iou.item(), rel_vel.item(), 1.0, is_p2p, is_p2o, is_o2o, is_p2w, is_p2b, is_w2o])
                        else:
                            # Legacy 4 kinematic + 3 relational = 7 dims
                            edge_attrs.append([d.item(), iou.item(), rel_vel.item(), 1.0, is_p2p, is_p2o, is_o2o])
                    else:
                        # Non-semantic 4 dims: [dist, iou, rel_vel, persistence]
                        edge_attrs.append([d.item(), iou.item(), rel_vel.item(), 1.0])

                    edges_src.append(i)
                    edges_dst.append(j)

        if not edges_src:
            for i in range(n):
                edges_src.append(i)
                edges_dst.append(i)
                if self.use_semantic and sem_types is not None:
                    ti = int(sem_types[i].item() if hasattr(sem_types[i], 'item') else sem_types[i])
                    is_p = 1.0 if ti == SEMANTIC_TYPE_PERSON else 0.0
                    if self.semantic_schema_version == "threat5":
                        is_w = 1.0 if ti == SEMANTIC_TYPE_WEAPON else 0.0
                        edge_attrs.append([0.0, 1.0, 0.0, 1.0, is_p, 0.0, 1.0 - is_p, 0.0, 0.0, is_w])
                    else:
                        edge_attrs.append([0.0, 1.0, 0.0, 1.0, is_p, 0.0, 1.0 - is_p])
                else:
                    edge_attrs.append([0.0, 1.0, 0.0, 1.0])

        src = torch.tensor(edges_src, device=frame.device, dtype=torch.long)
        dst = torch.tensor(edges_dst, device=frame.device, dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, device=frame.device, dtype=torch.float32)
        return node_feats, (src, dst), edge_attr, boxes


class EdgeGAT(nn.Module):
    """
    Edge-Conditioned Multi-Head Graph Attention Network.
    Fully vectorized with PyTorch scatter operations for fast training and inference.
    """
    def __init__(self, in_dim: int = 6, edge_dim: int = 4, out_dim: int = 128, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.attn = nn.Parameter(torch.randn(heads, self.head_dim * 2 + self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor], edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 0:
            return torch.zeros((0, self.heads * self.head_dim), device=x.device, dtype=x.dtype)
        h = self.proj(x).view(n, self.heads, self.head_dim)
        src, dst = edges
        e_feat = self.edge_proj(edge_attr).view(-1, self.heads, self.head_dim)

        h_src = h[src]
        h_dst = h[dst]
        combined = torch.cat([h_src, h_dst, e_feat], dim=-1)
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

        msg = (h_src + e_feat) * alpha
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
