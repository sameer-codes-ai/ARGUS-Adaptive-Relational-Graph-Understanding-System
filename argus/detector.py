"""
Semantic Object Detector & RoI Feature Extractor.

Implements Module 3 of the ARGUS research design:
  - Detects bounding boxes with semantic classes (Person, Vehicle, Object).
  - Extracts RoI-aligned visual appearance features for every graph entity
    from backbone feature maps using bilinear grid sampling.
"""

from __future__ import annotations
from typing import Optional, Tuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ARGUS Semantic Types & COCO Mapping ───────────────────────────────
SEMANTIC_TYPE_PERSON = 0
SEMANTIC_TYPE_VEHICLE = 1
SEMANTIC_TYPE_BAG = 2
SEMANTIC_TYPE_WEAPON = 3
SEMANTIC_TYPE_OTHER = 4

NUM_SEMANTIC_CLASSES = 5

SEMANTIC_CLASS_NAMES = {
    SEMANTIC_TYPE_PERSON: "person",
    SEMANTIC_TYPE_VEHICLE: "vehicle",
    SEMANTIC_TYPE_BAG: "bag",
    SEMANTIC_TYPE_WEAPON: "weapon",
    SEMANTIC_TYPE_OTHER: "other object",
}

# Single explicit COCO-to-ARGUS semantic mapping:
# 0 -> Person
# 1..8 -> Vehicle (bicycle, car, motorcycle, airplane, bus, train, truck, boat)
# 24, 26, 28 -> Bag (backpack, handbag, suitcase)
# 43 -> Weapon (knife)
# All other COCO classes default to 4 (Other object).
COCO_TO_ARGUS: dict[int, int] = {
    0: SEMANTIC_TYPE_PERSON,
    1: SEMANTIC_TYPE_VEHICLE,
    2: SEMANTIC_TYPE_VEHICLE,
    3: SEMANTIC_TYPE_VEHICLE,
    4: SEMANTIC_TYPE_VEHICLE,
    5: SEMANTIC_TYPE_VEHICLE,
    6: SEMANTIC_TYPE_VEHICLE,
    7: SEMANTIC_TYPE_VEHICLE,
    8: SEMANTIC_TYPE_VEHICLE,
    24: SEMANTIC_TYPE_BAG,
    26: SEMANTIC_TYPE_BAG,
    28: SEMANTIC_TYPE_BAG,
    43: SEMANTIC_TYPE_WEAPON,
}

PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8}
BAG_CLASSES = {24, 26, 28}
WEAPON_CLASSES = {43}


def _semantic_type(cls_id: int) -> int:
    """Map COCO class index to ARGUS semantic type:
    0=person, 1=vehicle, 2=bag, 3=weapon, 4=other object.
    Unknown COCO classes map to 4 (other object).
    """
    return COCO_TO_ARGUS.get(int(cls_id), SEMANTIC_TYPE_OTHER)


class DetectionResult:
    """Container for per-frame detection outputs."""
    __slots__ = ("boxes", "scores", "class_ids", "semantic_types", "detector_impl")

    def __init__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        semantic_types: torch.Tensor,
        detector_impl: str = "yolo",
    ):
        self.boxes = boxes                      # (N, 4) normalised xyxy in [0, 1]
        self.scores = scores                    # (N,) detector confidence in [0, 1]
        self.class_ids = class_ids              # (N,) original COCO class index
        self.semantic_types = semantic_types    # (N,) 0=person, 1=vehicle, 2=bag, 3=weapon, 4=other object
        self.detector_impl = detector_impl      # "yolo", "heuristic", or "grid"


class RoIFeatureExtractor(nn.Module):
    """
    Extracts fixed-dimensional appearance features for bounding boxes from
    a feature map using pure-PyTorch bilinear grid sampling.
    """
    def __init__(self, in_channels: int, out_dim: int = 64, pool_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.pool_size = pool_size
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, feature_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (C, H, W) or (1, C, H, W)
            boxes: (N, 4) in normalised [0, 1] (x1, y1, x2, y2)
        Returns:
            roi_features: (N, out_dim)
        """
        if feature_map.ndim == 3:
            feature_map = feature_map.unsqueeze(0)  # (1, C, H, W)

        n = boxes.shape[0]
        if n == 0:
            return torch.zeros((0, self.out_dim), device=feature_map.device)

        # Build grid for each box: map [x1, y1, x2, y2] in [0, 1] to [-1, 1]
        # x_norm = 2 * x - 1
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Generate sample points for (pool_size, pool_size)
        steps = torch.linspace(0, 1, self.pool_size, device=boxes.device)
        # grid shape: (N, pool_size, pool_size, 2)
        gx = x1[:, None, None] + (x2 - x1)[:, None, None] * steps[None, None, :]
        gy = y1[:, None, None] + (y2 - y1)[:, None, None] * steps[None, :, None]
        gx = gx.expand(n, self.pool_size, self.pool_size)
        gy = gy.expand(n, self.pool_size, self.pool_size)

        grid = torch.stack([gx * 2.0 - 1.0, gy * 2.0 - 1.0], dim=-1)  # (N, P, P, 2)

        # Expand feature_map to (N, C, H, W)
        f_expanded = feature_map.expand(n, -1, -1, -1)
        sampled = F.grid_sample(f_expanded, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # sampled: (N, C, P, P)
        pooled = sampled.mean(dim=[-2, -1])  # (N, C)

        return self.proj(pooled)  # (N, out_dim)


class SemanticDetector(nn.Module):
    """
    Semantic Object Detector with YOLOv11 and heuristic GridDetector fallback.
    """
    def __init__(
        self,
        confidence: float = 0.3,
        max_objects: int = 16,
        model_name: str = "yolo11n.pt",
        allow_fallback: bool = True,
    ):
        super().__init__()
        self.confidence = confidence
        self.max_objects = max_objects
        self.model_name = model_name
        self.yolo = None
        self._fallback = None

        if model_name in {"heuristic", "grid"}:
            from .graph import GridDetector
            self._fallback = GridDetector(max_objects)
            return

        try:
            from ultralytics import YOLO
            # Ultralytics' high-level YOLO wrapper exposes ``train`` as a
            # training entry point rather than ``nn.Module.train(mode)``.
            # Registering it as a PyTorch child module therefore makes the
            # parent ARGUS ``model.train()`` call crash. Keep the frozen
            # inference wrapper as a plain attribute; its underlying weights
            # are external pretrained detector state, not ARGUS parameters.
            object.__setattr__(self, "yolo", YOLO(model_name))
            self.yolo.fuse()
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "eval"):
                self.yolo.model.eval()
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Unable to initialize the requested YOLO semantic detector. Install ultralytics "
                    "and make its weights available, or explicitly enable component fallbacks."
                ) from exc
            warnings.warn(
                f"YOLO semantic detection is unavailable; using heuristic proposals: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            from .graph import GridDetector
            self._fallback = GridDetector(max_objects)

    @property
    def has_yolo(self) -> bool:
        return self.yolo is not None

    @torch.no_grad()
    def detect_batch(self, frames: torch.Tensor) -> list[DetectionResult]:
        """
        Batched detection across N frames in a single GPU pass.
        Args:
            frames: (N, 3, H, W) normalized [0, 1] tensor on device.
        Returns:
            list of N DetectionResult objects.
        """
        if frames.ndim == 3:
            return [self.forward(frames)]
        n, _, h, w = frames.shape
        if n == 0:
            return []
        if self.yolo is not None:
            img = frames.float().clamp(0.0, 1.0)
            results = self.yolo(img, verbose=False, conf=self.confidence)
            batch_results = []
            for i in range(n):
                res = results[i]
                if len(res.boxes) == 0:
                    batch_results.append(self._empty(frames.device))
                    continue
                det = res.boxes
                boxes_px = det.xyxy.float()
                scores = det.conf.float()
                cids = det.cls.long()

                boxes_norm = boxes_px.clone()
                boxes_norm[:, [0, 2]] /= w
                boxes_norm[:, [1, 3]] /= h

                sem = torch.tensor(
                    [_semantic_type(int(c)) for c in cids],
                    device=frames.device,
                    dtype=torch.long
                )

                if len(boxes_norm) > self.max_objects:
                    topk = scores.topk(self.max_objects).indices
                    boxes_norm, scores, cids, sem = (
                        boxes_norm[topk], scores[topk], cids[topk], sem[topk]
                    )

                batch_results.append(DetectionResult(
                    boxes=boxes_norm.to(frames.device),
                    scores=scores.to(frames.device),
                    class_ids=cids.to(frames.device),
                    semantic_types=sem.to(frames.device),
                    detector_impl="yolo",
                ))
            return batch_results
        return [self._fallback_detect(frames[i], h, w) for i in range(n)]

    @torch.no_grad()
    def forward(self, frame: torch.Tensor) -> DetectionResult:
        """
        Args:
            frame: (3, H, W) normalized [0, 1] tensor.
        Returns:
            DetectionResult
        """
        h, w = frame.shape[-2:]
        if self.yolo is not None:
            return self._yolo_detect(frame, h, w)
        return self._fallback_detect(frame, h, w)

    def _yolo_detect(self, frame: torch.Tensor, h: int, w: int) -> DetectionResult:
        # Ultralytics accepts BCHW float tensors normalized to [0, 1]. Passing
        # byte-like values in [0, 255] makes it renormalize every frame and
        # emits a warning for every detector call.
        img = frame.unsqueeze(0).float().clamp(0.0, 1.0)
        results = self.yolo(img, verbose=False, conf=self.confidence)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return self._empty(frame.device)

        det = results[0].boxes
        boxes_px = det.xyxy.float()
        scores = det.conf.float()
        cids = det.cls.long()

        boxes_norm = boxes_px.clone()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h

        sem = torch.tensor(
            [_semantic_type(int(c)) for c in cids],
            device=frame.device,
            dtype=torch.long
        )

        if len(boxes_norm) > self.max_objects:
            topk = scores.topk(self.max_objects).indices
            boxes_norm, scores, cids, sem = (
                boxes_norm[topk], scores[topk], cids[topk], sem[topk]
            )

        return DetectionResult(
            boxes=boxes_norm.to(frame.device),
            scores=scores.to(frame.device),
            class_ids=cids.to(frame.device),
            semantic_types=sem.to(frame.device),
            detector_impl="yolo",
        )

    def _fallback_detect(self, frame: torch.Tensor, h: int, w: int) -> DetectionResult:
        boxes = self._fallback(frame)
        n = len(boxes)

        # Heuristic semantic type tagging:
        # Tall boxes (height > 1.3 * width) -> likely Person (0)
        # Broad or very large boxes -> Vehicle (1)
        # All others -> Other object (4)
        # NEVER emit fake Weapon (3) or Bag (2) in heuristic fallback mode!
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        aspect_ratio = heights / (widths + 1e-6)

        sem = torch.full((n,), SEMANTIC_TYPE_OTHER, device=frame.device, dtype=torch.long)  # default: Other object (4)
        sem[widths * heights > 0.25] = SEMANTIC_TYPE_VEHICLE   # Large vehicle (1)
        sem[aspect_ratio > 1.3] = SEMANTIC_TYPE_PERSON        # Person (tall boxes override) (0)

        impl = "grid" if self.model_name in {"heuristic", "grid"} and self.yolo is None else "heuristic"
        return DetectionResult(
            boxes=boxes,
            scores=torch.ones(n, device=frame.device),
            class_ids=sem,
            semantic_types=sem,
            detector_impl=impl,
        )

    def _empty(self, device: torch.device) -> DetectionResult:
        impl = "yolo" if self.yolo is not None else ("grid" if self.model_name in {"heuristic", "grid"} else "heuristic")
        return DetectionResult(
            boxes=torch.empty((0, 4), device=device),
            scores=torch.empty((0,), device=device),
            class_ids=torch.empty((0,), device=device, dtype=torch.long),
            semantic_types=torch.empty((0,), device=device, dtype=torch.long),
            detector_impl=impl,
        )
