"""
ARGUS Matched Baseline Architectures.
Provides compute-matched surveillance baselines:
  - ConvNeXtMIL: 2D Spatial Backbone + Temporal Pooling + Top-k MIL
  - CausalTSM: Causal Temporal Shift Module (past-only shift)
  - RTFMLite: Multi-Scale Temporal Network (MTN) + Feature Magnitude Ranking
"""

from __future__ import annotations
from typing import Optional
import torch.nn as nn

from argus.config import ARGUSConfig
from .convnext_mil import ConvNeXtMIL
from .causal_tsm import CausalTSM
from .rtfm_lite import RTFMLite


def build_model(model_type: str = "argus", config: Optional[ARGUSConfig] = None) -> nn.Module:
    """
    Factory function to instantiate ARGUS or one of the compute-matched baselines.
    """
    mtype = model_type.lower().strip()
    if mtype == "argus":
        from argus.model import ARGUS
        return ARGUS(config=config)
    elif mtype in {"convnext_mil", "convnext", "mil"}:
        return ConvNeXtMIL(config=config)
    elif mtype in {"causal_tsm", "tsm"}:
        return CausalTSM(config=config)
    elif mtype in {"rtfm_lite", "rtfm"}:
        return RTFMLite(config=config)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Supported types: ['argus', 'convnext_mil', 'causal_tsm', 'rtfm_lite']"
        )


__all__ = [
    "ConvNeXtMIL",
    "CausalTSM",
    "RTFMLite",
    "build_model",
]
