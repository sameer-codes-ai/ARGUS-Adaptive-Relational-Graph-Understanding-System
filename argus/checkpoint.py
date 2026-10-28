"""Validated, safe checkpoint loading helpers for ARGUS."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, map_location: Any = "cpu", *, allow_unsafe: bool = False):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe:
            raise ValueError(
                f"Checkpoint could not be loaded in safe weights-only mode: {checkpoint_path}. "
                "Use the explicit unsafe compatibility option only for a trusted legacy file."
            ) from safe_error
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must be a state-dict mapping or a mapping containing a 'model' key")
    return state


def model_state_dict(checkpoint: Mapping):
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint does not contain a non-empty model state dictionary")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Model state dictionary keys must be strings")
    return state_dict


def validate_component_metadata(checkpoint: Mapping, model) -> None:
    saved = checkpoint.get("components")
    if not saved:
        return
    current = model.component_metadata()
    keys = ("backbone_implementation", "backbone_name", "detector_implementation", "detector_model")
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in keys
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        details = ", ".join(f"{key}: saved={old!r}, current={new!r}" for key, (old, new) in mismatches.items())
        raise RuntimeError(f"Checkpoint component mismatch: {details}")
