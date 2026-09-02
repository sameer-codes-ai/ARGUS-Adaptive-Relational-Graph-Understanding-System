from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class ARGUSConfig:
    image_size: int = 96
    patch_size: int = 16
    dim: int = 192
    graph_dim: int = 128
    max_objects: int = 12
    temporal_layers: int = 3
    heads: int = 4
    keep_ratio: float = 0.5
    memory_size: int = 16
    sampling_threshold: float = 0.12
    max_frame_gap: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ARGUSConfig':
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'ARGUSConfig':
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        model_cfg = data.get('model', data)
        return cls.from_dict(model_cfg)
