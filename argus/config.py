from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class ARGUSConfig:
    # ── Spatial dimensions ──
    image_size: int = 96
    patch_size: int = 16
    dim: int = 192
    graph_dim: int = 128
    max_objects: int = 12
    # ── Temporal transformer ──
    temporal_layers: int = 3
    heads: int = 4
    keep_ratio: float = 0.5
    memory_size: int = 16
    # ── Adaptive sampling ──
    sampling_threshold: float = 0.12
    max_frame_gap: int = 4
    # ── Outer layer: Spatial Backbone ──
    use_pretrained_backbone: bool = False
    backbone_name: str = 'convnext_tiny'
    backbone_frozen: bool = True
    # ── Outer layer: Semantic Detection ──
    use_semantic_detector: bool = False
    detector_name: str = 'yolo11n.pt'
    detector_confidence: float = 0.3
    # ── Outer layer: RoI node features ──
    node_visual_dim: int = 0  # Set > 0 to append RoI features to graph nodes
    # Optional components must never silently change the architecture in production.
    allow_component_fallbacks: bool = False
    backbone_pretrained: bool = True
    tracking_iou_threshold: float = 0.1

    def validate(self) -> 'ARGUSConfig':
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0 or self.patch_size > self.image_size:
            raise ValueError("patch_size must be positive and no larger than image_size")
        if self.dim <= 0 or self.graph_dim <= 0:
            raise ValueError("dim and graph_dim must be positive")
        if self.heads <= 0 or self.dim % self.heads or self.graph_dim % self.heads:
            raise ValueError("dim and graph_dim must both be divisible by heads")
        if self.temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if self.max_objects <= 0 or self.max_frame_gap <= 0:
            raise ValueError("max_objects and max_frame_gap must be positive")
        if self.sampling_threshold < 0.0:
            raise ValueError("sampling_threshold cannot be negative")
        if not 0.0 <= self.detector_confidence <= 1.0:
            raise ValueError("detector_confidence must be in [0, 1]")
        if not self.detector_name:
            raise ValueError("detector_name cannot be empty")
        if self.node_visual_dim < 0:
            raise ValueError("node_visual_dim cannot be negative")
        if not 0.0 <= self.tracking_iou_threshold <= 1.0:
            raise ValueError("tracking_iou_threshold must be in [0, 1]")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, strict: bool = True) -> 'ARGUSConfig':
        if not isinstance(d, dict):
            raise TypeError("model configuration must be a mapping")
        valid_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(d) - valid_keys
        if strict and unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered).validate()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'ARGUSConfig':
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        model_cfg = data.get('model', data)
        return cls.from_dict(model_cfg, strict=True)
