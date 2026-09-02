import torch
import torch.nn as nn
import torch.nn.functional as F

class GridDetector:
    def __init__(self, max_objects=12):
        self.max_objects = max_objects

    @torch.no_grad()
    def __call__(self, frame):
        h, w = frame.shape[-2:]
        energy = frame.mean(0)
        cell = max(8, min(h, w) // 6)
        candidates = []
        for y in range(0, h - cell + 1, cell):
            for x in range(0, w - cell + 1, cell):
                candidates.append((energy[y:y + cell, x:x + cell].var().item(), x, y, x + cell, y + cell))
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
    def __init__(self, max_objects=12, dist_threshold=0.6):
        self.detector = GridDetector(max_objects)
        self.dist_threshold = dist_threshold

    def build(self, frame, prev_boxes=None):
        boxes = self.detector(frame)
        n = len(boxes)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)

        if prev_boxes is not None and len(prev_boxes) == n:
            prev_centers = (prev_boxes[:, :2] + prev_boxes[:, 2:]) / 2.0
            velocity = centers - prev_centers
        else:
            velocity = torch.zeros_like(centers)

        node_feats = torch.cat([centers, sizes, velocity], dim=-1)

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
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_attrs.append([d.item(), iou.item(), rel_vel.item(), 1.0])

        if not edges_src:
            for i in range(n):
                edges_src.append(i)
                edges_dst.append(i)
                edge_attrs.append([0.0, 1.0, 0.0, 1.0])

        src = torch.tensor(edges_src, device=frame.device, dtype=torch.long)
        dst = torch.tensor(edges_dst, device=frame.device, dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, device=frame.device, dtype=torch.float32)
        return node_feats, (src, dst), edge_attr, boxes


class EdgeGAT(nn.Module):
    def __init__(self, in_dim=6, edge_dim=4, out_dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.attn = nn.Parameter(torch.randn(heads, self.head_dim * 2 + self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edges, edge_attr):
        n = x.shape[0]
        h = self.proj(x).view(n, self.heads, self.head_dim)
        src, dst = edges
        e_feat = self.edge_proj(edge_attr).view(-1, self.heads, self.head_dim)

        h_src = h[src]
        h_dst = h[dst]
        combined = torch.cat([h_src, h_dst, e_feat], dim=-1)
        logits = (combined * self.attn.unsqueeze(0)).sum(dim=-1)
        logits = F.leaky_relu(logits, 0.2)

        out = torch.zeros_like(h)
        for i in range(n):
            mask = (dst == i)
            if mask.any():
                alpha = F.softmax(logits[mask], dim=0).unsqueeze(-1)
                msg = (h_src[mask] + e_feat[mask]) * alpha
                out[i] = msg.sum(dim=0)
            else:
                out[i] = h[i]

        out = out.view(n, -1)
        return self.out_proj(out)
