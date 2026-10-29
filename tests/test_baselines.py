"""
Unit tests for Phase 2: Compute-Matched Real Surveillance Baselines.
Verifies:
  - ConvNeXtMIL architecture, forward pass, shape contracts, and backward pass
  - CausalTSM architecture, causality invariance, streaming buffer, and loss
  - RTFMLite architecture, MTN multi-scale features, magnitude ranking loss
  - build_model factory function and error handling
"""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

from argus.config import ARGUSConfig
from argus.baselines import build_model, ConvNeXtMIL, CausalTSM, RTFMLite
from argus.losses import compute_model_loss, rtfm_magnitude_loss


@pytest.fixture
def fast_cfg():
    return ARGUSConfig(
        dim=64,
        image_size=64,
        patch_size=16,
        use_pretrained_backbone=False,
    )


def test_build_model_factory(fast_cfg):
    m_argus = build_model("argus", config=fast_cfg)
    assert m_argus.__class__.__name__ == "ARGUS"

    m_mil = build_model("convnext_mil", config=fast_cfg)
    assert isinstance(m_mil, ConvNeXtMIL)

    m_tsm = build_model("causal_tsm", config=fast_cfg)
    assert isinstance(m_tsm, CausalTSM)

    m_rtfm = build_model("rtfm_lite", config=fast_cfg)
    assert isinstance(m_rtfm, RTFMLite)

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model("unknown_model", config=fast_cfg)


def test_convnext_mil_forward_and_loss(fast_cfg):
    model = ConvNeXtMIL(config=fast_cfg)
    model.train()
    x = torch.rand(2, 8, 3, 64, 64)
    y = torch.tensor([1.0, 0.0])

    out = model(x)
    assert out.scores.shape == (2, 8)
    assert torch.all(out.scores >= 0.0) and torch.all(out.scores <= 1.0)
    assert out.valid_mask.shape == (2, 8)

    loss, loss_dict = compute_model_loss("convnext_mil", out, y, model=model)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "bce" in loss_dict and "mil" in loss_dict

    loss.backward()
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert has_grad


def test_causal_tsm_forward_and_streaming(fast_cfg):
    model = CausalTSM(config=fast_cfg)
    model.eval()

    # 1. Non-streaming forward pass
    x = torch.rand(2, 8, 3, 64, 64)
    out = model(x, stream=False)
    assert out.scores.shape == (2, 8)
    assert torch.all(out.scores >= 0.0) and torch.all(out.scores <= 1.0)

    # 2. Streaming causality verification:
    # Frame 0 prediction must NOT change if future frames are altered!
    clip1 = torch.rand(1, 4, 3, 64, 64)
    clip2 = clip1.clone()
    clip2[:, 2:] = torch.rand(1, 2, 3, 64, 64)  # modify frames 2 and 3

    model.reset("test_stream")
    out1 = model(clip1, stream=True, stream_id="test_stream")

    model.reset("test_stream")
    out2 = model(clip2, stream=True, stream_id="test_stream")

    # Frame 0 and Frame 1 should be identical because shift is strictly past-only
    assert torch.allclose(out1.scores[:, :2], out2.scores[:, :2], atol=1e-5)


def test_rtfm_lite_forward_and_magnitude_loss(fast_cfg):
    model = RTFMLite(config=fast_cfg)
    model.train()
    x = torch.rand(2, 8, 3, 64, 64)
    y = torch.tensor([1.0, 0.0])

    out = model(x)
    assert out.scores.shape == (2, 8)
    assert hasattr(model, "last_temporal_features")
    assert model.last_temporal_features is not None

    mags = model.compute_magnitudes(model.last_temporal_features)
    assert mags.shape == (2, 8)

    loss, loss_dict = compute_model_loss("rtfm_lite", out, y, model=model)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "mag_loss" in loss_dict

    loss.backward()
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert has_grad
