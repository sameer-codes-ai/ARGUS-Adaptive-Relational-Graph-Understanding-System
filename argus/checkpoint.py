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


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint's semantic schema or architecture is incompatible with the model."""
    pass


def validate_component_metadata(checkpoint: Mapping, model, *, allow_partial: bool = False) -> None:
    saved = checkpoint.get("components")

    # Check semantic schema version and EdgeGAT input dimensions FIRST
    saved_schema = (
        checkpoint.get("semantic_schema_version")
        or (saved.get("semantic_schema_version") if saved else None)
        or checkpoint.get("config", {}).get("semantic_schema_version")
    )
    current_schema = getattr(getattr(model, "cfg", None), "semantic_schema_version", "threat5")

    # Inspect state dictionary for EdgeGAT dimension compatibility
    state_dict = None
    try:
        state_dict = model_state_dict(checkpoint)
    except Exception:
        pass

    mismatch_detail = None
    if saved_schema and saved_schema != current_schema:
        mismatch_detail = f"semantic schema version mismatch: saved={saved_schema!r}, model={current_schema!r}"
    elif state_dict and "edge_gat.proj.weight" in state_dict and hasattr(model, "edge_gat"):
        saved_gat_in = state_dict["edge_gat.proj.weight"].shape[1]
        expected_gat_in = model.edge_gat.proj.in_features
        if saved_gat_in != expected_gat_in:
            detected_schema = "legacy3" if saved_gat_in == 10 else f"dim_{saved_gat_in}"
            mismatch_detail = (
                f"EdgeGAT input dimension mismatch: saved weights have in_dim={saved_gat_in} ({detected_schema}), "
                f"but model expects in_dim={expected_gat_in} ({current_schema})"
            )

    if mismatch_detail:
        if not allow_partial:
            raise IncompatibleCheckpointError(
                f"Incompatible checkpoint: {mismatch_detail}. "
                "Models with the expanded 5-class semantic schema ('threat5') cannot load legacy 3-class ('legacy3') "
                "graph-attention weights directly. Rejecting incompatible checkpoint. Retrain the model or use "
                "partial initialization for unchanged modules."
            )

    if saved:
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


def load_partial_checkpoint(model: torch.nn.Module, checkpoint: Mapping) -> dict[str, list[str]]:
    """
    Allow partial initialization from an old or shape-divergent checkpoint.
    Loads matching parameters for unchanged modules:
      - backbone / patch_proj
      - fusion
      - pruner
      - temporal_encoder
      - memory
      - head (if shape matches)
    Initializes changed graph-attention layers (EdgeGAT) fresh.
    """
    state_dict = model_state_dict(checkpoint)
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    to_load = {}
    for k, v in state_dict.items():
        if k.startswith("edge_gat."):
            skipped_keys.append(k)
            continue
        if k in model_state and model_state[k].shape == v.shape:
            to_load[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(to_load, strict=False)
    return {"loaded": loaded_keys, "skipped": skipped_keys}
