"""
Unit tests for Phase 3: Controlled Component Ablation Suite.

Verifies:
  - All 6 YAML configs in configs/ablations/ load and validate cleanly
  - Ablation 1 (No Graph): Token bypass without detector or EdgeGAT
  - Ablation 2 (No Edge Attributes): Standard GAT without edge attributes
  - Ablation 3 (No RoI Features): Graph nodes without visual crops (dim=0)
  - Ablation 4 (No Pruning): Full token retention (keep_ratio=1.0)
  - Ablation 5 (Uniform Sampling): Bypasses motion entropy for uniform indexing
  - Ablation 6 (No Causal Memory): Memory size 0 bypass
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
from argus.model import ARGUS
from argus.graph import EdgeGAT
from argus.sampling import AdaptiveSampler
from argus.modules import GraphGuidedPruner, TemporalMemory


def test_ablation_configs_load_and_validate():
    configs_dir = PROJECT_ROOT / "configs" / "ablations"
    expected_configs = [
        "ablation_1_no_graph.yaml",
        "ablation_2_no_edge_attrs.yaml",
        "ablation_3_no_roi_features.yaml",
        "ablation_4_no_pruning.yaml",
        "ablation_5_uniform_sampling.yaml",
        "ablation_6_no_causal_memory.yaml",
    ]
    for cfg_name in expected_configs:
        cfg_path = configs_dir / cfg_name
        assert cfg_path.is_file(), f"Missing config file: {cfg_name}"
        loaded = ARGUSConfig.from_yaml(cfg_path)
        assert isinstance(loaded, ARGUSConfig)


def test_ablation_1_no_graph():
    cfg = ARGUSConfig(
        dim=32,
        heads=2,
        temporal_layers=1,
        image_size=32,
        patch_size=16,
        use_pretrained_backbone=False,
        use_graph=False,
        use_semantic_detector=False,
        node_visual_dim=0,
    )
    model = ARGUS(cfg).eval()
    assert model.graph_builder is None
    assert model.edge_gat is None
    assert model.fusion is None

    x = torch.rand(1, 4, 3, 32, 32)
    out = model(x)
    assert out.scores.shape == (1, 4)
    assert torch.all(out.scores >= 0.0) and torch.all(out.scores <= 1.0)


def test_ablation_2_no_edge_attrs():
    gat = EdgeGAT(in_dim=6, edge_dim=4, out_dim=32, heads=2, use_edge_attrs=False)
    assert gat.edge_proj is None

    # 3 nodes, 2 edges
    x = torch.rand(3, 6)
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 2], dtype=torch.long)
    out = gat(x, (src, dst), edge_attr=None)
    assert out.shape == (3, 32)


def test_ablation_3_no_roi_features():
    cfg = ARGUSConfig(
        dim=32,
        heads=2,
        temporal_layers=1,
        image_size=32,
        patch_size=16,
        use_pretrained_backbone=False,
        use_semantic_detector=False,
        node_visual_dim=0,
    )
    model = ARGUS(cfg).eval()
    assert model.roi_extractor is None

    x = torch.rand(1, 4, 3, 32, 32)
    out = model(x)
    assert out.scores.shape == (1, 4)


def test_ablation_4_no_pruning():
    pruner = GraphGuidedPruner(visual_dim=32, keep_ratio=1.0)
    tokens = torch.rand(2, 16, 32)
    pruned, scores, indices = pruner(tokens)
    assert pruned.shape == (2, 16, 32)  # All 16 tokens kept!


def test_ablation_5_uniform_sampling():
    sampler = AdaptiveSampler(threshold=0.12, max_gap=4, uniform_sampling=True)
    frames = torch.rand(2, 16, 3, 32, 32)
    selected, scores = sampler.select(frames)
    assert len(selected) == 2
    assert selected[0].tolist() == list(range(16))
    assert selected[1].tolist() == list(range(16))


def test_ablation_6_no_causal_memory():
    mem = TemporalMemory(dim=32, size=0)
    latest = torch.rand(1, 4, 32)
    enriched = mem(latest, update=True, stream_id="stream_0")
    assert enriched.shape == (1, 4, 32)
    # Stream bank should not store history
    assert mem._stream_banks == {}

def test_ablation_models_state_dict_and_metadata():
    configs_dir = PROJECT_ROOT / "configs" / "ablations"
    for cfg_name in [
        "ablation_1_no_graph.yaml",
        "ablation_2_no_edge_attrs.yaml",
        "ablation_3_no_roi_features.yaml",
        "ablation_4_no_pruning.yaml",
        "ablation_5_uniform_sampling.yaml",
        "ablation_6_no_causal_memory.yaml",
    ]:
        cfg = ARGUSConfig.from_yaml(configs_dir / cfg_name)
        cfg.dim = 32
        cfg.heads = 2
        cfg.temporal_layers = 1
        cfg.image_size = 32
        cfg.patch_size = 16
        cfg.use_pretrained_backbone = False
        cfg.use_semantic_detector = False
        model = ARGUS(cfg).eval()
        meta = model.component_metadata()
        assert isinstance(meta, dict)
        sd = model.state_dict()
        assert len(sd) > 0


def test_ablation_backward_passes():
    # Verify backward gradient flow on Ablation 1 (no graph) and Ablation 2 (no edge attrs)
    for use_graph in [False, True]:
        cfg = ARGUSConfig(
            dim=32,
            heads=2,
            temporal_layers=1,
            image_size=32,
            patch_size=16,
            use_pretrained_backbone=False,
            use_semantic_detector=False,
            node_visual_dim=0,
            use_graph=use_graph,
            use_edge_attrs=False if use_graph else True,
        )
        model = ARGUS(cfg).train()
        x = torch.rand(2, 4, 3, 32, 32)
        out = model(x)
        loss = out.scores.sum()
        loss.backward()
        has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
        assert has_grad
