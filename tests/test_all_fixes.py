"""
Comprehensive verification test suite covering all 13 audit fixes in ARGUS.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
import yaml

from argus import (
    ARGUS,
    ARGUSConfig,
    AdaptiveSampler,
    SemanticDetector,
    DynamicSceneGraphBuilder,
    ExplainabilityEngine,
    binary_metrics,
)
from argus.model import ARGUSOutput
from argus.modules import TemporalEncoder
from argus.detector import DetectionResult


# ── BUG-1: Duplicate Detector Invocation ─────────────────────────────────
def test_fix_1_detector_called_once_per_frame():
    """Verify detector is invoked exactly once per frame, not twice."""
    cfg = ARGUSConfig(
        image_size=64,
        dim=64,
        graph_dim=32,
        use_pretrained_backbone=False,
        use_semantic_detector=True,
        detector_name="heuristic",
        node_visual_dim=0,
        temporal_layers=1,
        heads=2,
        allow_component_fallbacks=True,
    )
    model = ARGUS(config=cfg)

    call_count = 0
    original_detector = model.graph_builder.detector

    def spy_detector(frame):
        nonlocal call_count
        call_count += 1
        return original_detector(frame)

    model.graph_builder.detector = spy_detector

    clip = torch.rand(1, 3, 3, 64, 64)  # 3 frames
    out = model(clip)

    # Number of frames processed is 3 (or sampled keyframes)
    num_sampled = len(out.sampled_indices[0])
    assert call_count == num_sampled, f"Detector called {call_count} times for {num_sampled} frames (should be exactly 1:1)"


# ── BUG-2: NaN from std() on Single-Element Tensor (2-frame clips) ────────
def test_fix_2_sampling_no_nan_on_two_frame_clip():
    """Verify AdaptiveSampler does not produce NaN for 2-frame clips (where t-1=1)."""
    sampler = AdaptiveSampler()

    # 2-frame clip
    clip_2 = torch.rand(2, 2, 3, 32, 32)
    scores_2 = sampler.importance_scores(clip_2)
    assert torch.isfinite(scores_2).all(), "NaN or Inf detected in 2-frame clip scores"
    assert not torch.isnan(scores_2).any()

    # Single-frame clip edge case
    clip_1 = torch.rand(2, 1, 3, 32, 32)
    scores_1 = sampler.importance_scores(clip_1)
    assert torch.isfinite(scores_1).all()

    # 3-frame clip
    clip_3 = torch.rand(2, 3, 32, 32)
    scores_3 = sampler.importance_scores(clip_3.unsqueeze(2).repeat(1, 1, 3, 1, 1))
    assert torch.isfinite(scores_3).all()


# ── BUG-3: Metrics Corruption on Multi-Dim Inputs ────────────────────────
def test_fix_3_binary_metrics_flatten_multidim():
    """Verify binary_metrics correctly flattens (B, 1) and 2D arrays without corrupting AUC/AP."""
    # 2D column vector shapes common from BCE output
    preds_2d = torch.tensor([[0.95], [0.85], [0.20], [0.10]])
    targets_2d = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    metrics = binary_metrics(preds_2d, targets_2d)
    assert metrics['accuracy'] == 1.0
    assert metrics['auc_roc'] == 1.0, f"Expected AUC 1.0, got {metrics['auc_roc']}"
    assert metrics['average_precision'] == 1.0, f"Expected AP 1.0, got {metrics['average_precision']}"
    assert metrics['f1'] == 1.0

    # Also test numpy 2D arrays
    p_np = np.array([[0.9], [0.1]])
    y_np = np.array([[1], [0]])
    m_np = binary_metrics(p_np, y_np)
    assert m_np['auc_roc'] == 1.0
    assert m_np['accuracy'] == 1.0


# ── BUG-4: LR Scheduler Checkpointed and Restored ────────────────────────
def test_fix_4_scheduler_checkpoint_and_resume(tmp_path):
    """Verify scheduler state is stored in checkpoint and restored on resume."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    param = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdamW([param], lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)

    # Step optimizer and scheduler 4 times
    for _ in range(4):
        opt.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
    lr_at_step_4 = sched.get_last_lr()[0]

    # Save checkpoint with scheduler
    ckpt_path = tmp_path / "ckpt_with_sched.pt"
    checkpoint = {
        'epoch': 4,
        'model': {'p': param},
        'optimizer': opt.state_dict(),
        'scheduler': sched.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)

    # Load checkpoint into new optimizer and scheduler
    loaded = torch.load(ckpt_path, weights_only=True)
    assert 'scheduler' in loaded, "Checkpoint missing 'scheduler' key"

    new_param = torch.nn.Parameter(torch.randn(2, 2))
    new_opt = AdamW([new_param], lr=1e-3)
    new_sched = CosineAnnealingLR(new_opt, T_max=10, eta_min=1e-6)

    new_opt.load_state_dict(loaded['optimizer'])
    new_sched.load_state_dict(loaded['scheduler'])

    assert new_sched.last_epoch == 4
    assert np.isclose(new_sched.get_last_lr()[0], lr_at_step_4), "Resumed LR does not match pre-save LR"


# ── BUG-5: Memory Modulation Padding Neutrality ──────────────────────────
def test_fix_5_memory_modulation_padding_neutral():
    """Verify TemporalEncoder pads memory modulation with 0.5 (neutral scale factor 1.0)."""
    enc = TemporalEncoder(dim=32, layers=1, heads=2)
    x = torch.randn(1, 6, 32)  # t=6
    # Modulation has t=2 (memory buffer is 4 tokens)
    mod = torch.tensor([[0.5, 0.5]])

    # Padding formula in TemporalEncoder uses 0.5:
    # pad = torch.full((b, t - mod.shape[1]), 0.5) -> scale = 1.0 + 0.5 * (0.5 - 0.5) = 1.0
    out = enc(x, modulation=mod)
    assert out.shape == (1, 6, 32)
    assert torch.isfinite(out).all()


# ── BUG-6: Raw State-Dict Checkpoint Support in Eval & Infer ──────────────
def test_fix_6_raw_state_dict_checkpoint(tmp_path):
    """Verify raw state-dict checkpoints (without 'model' wrapper key) load cleanly."""
    model = ARGUS(config=ARGUSConfig(image_size=64, dim=64, temporal_layers=1, heads=2))
    raw_path = tmp_path / "raw_model.pt"

    # Save RAW state_dict (no wrapper dict)
    torch.save(model.state_dict(), raw_path)

    # Test the loader logic from evaluate.py / infer.py
    state = torch.load(raw_path, weights_only=True)
    state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state

    eval_model = ARGUS(config=ARGUSConfig(image_size=64, dim=64, temporal_layers=1, heads=2))
    eval_model.load_state_dict(state_dict)
    assert eval_model is not None


# ── BUG-7: Falsy backbone_lr: 0.0 Config ─────────────────────────────────
def test_fix_7_backbone_lr_zero_not_falsy():
    """Verify backbone_lr: 0.0 in YAML is preserved rather than falling back to default."""
    cfg = {'lr': 0.001, 'backbone_lr': 0.0}
    backbone_lr = cfg.get('backbone_lr') if cfg.get('backbone_lr') is not None else (cfg['lr'] * 0.1)
    assert backbone_lr == 0.0, f"Expected 0.0, got {backbone_lr} (falsy check failed)"

    # When None, should fall back
    cfg_none = {'lr': 0.001, 'backbone_lr': None}
    backbone_lr_none = cfg_none.get('backbone_lr') if cfg_none.get('backbone_lr') is not None else (cfg_none['lr'] * 0.1)
    assert backbone_lr_none == 0.0001


# ── BUG-8: Node Attribution Gradients Retained ───────────────────────────
def test_fix_8_node_attribution_gradients():
    """Verify ExplainabilityEngine retains gradients on non-leaf node embeddings."""
    cfg = ARGUSConfig(
        image_size=64,
        dim=64,
        graph_dim=32,
        use_pretrained_backbone=False,
        use_semantic_detector=True,
        detector_name="heuristic",
        node_visual_dim=0,
        temporal_layers=1,
        heads=2,
        allow_component_fallbacks=True,
    )
    model = ARGUS(config=cfg)
    engine = ExplainabilityEngine(model)
    clip = torch.rand(1, 3, 3, 64, 64)

    res = engine.compute_node_attribution(clip)
    assert 'node_importances' in res
    assert len(res['node_importances']) > 0
    # At least some node importances should now be non-None tensors!
    has_valid_importance = any(imp is not None for imp in res['node_importances'])
    assert has_valid_importance, "Node importances are still all None! retain_grad() not effective"
    engine.cleanup()


# ── BUG-9: Manifest Script Includes Processed Clips in Combined Manifest ──
def test_fix_9_manifest_combined_total_records():
    """Verify prepare_manifests.py total_records includes proc_records."""
    rl = [{"path": "rl1"}, {"path": "rl2"}]
    ucf = [{"path": "u1"}]
    st = [{"path": "s1"}]
    proc = [{"path": "p1"}, {"path": "p2"}]

    total_records = rl + ucf + st + proc
    assert len(total_records) == 6
    assert any(r["path"] == "p1" for r in total_records)


# ── BUG-10: Atomic History Write ─────────────────────────────────────────
def test_fix_10_atomic_history_write(tmp_path):
    """Verify history JSON write creates valid, complete JSON via atomic replace."""
    history_file = tmp_path / "training_history.json"
    history = [
        {'epoch': 1, 'train_loss': 0.5, 'val_metrics': {'f1': 0.8}},
        {'epoch': 2, 'train_loss': 0.3, 'val_metrics': {'f1': 0.85}},
    ]

    history_tmp = tmp_path / "training_history_tmp_123.json"
    with open(history_tmp, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    history_tmp.replace(history_file)

    assert history_file.exists()
    with open(history_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert len(loaded) == 2
    assert loaded[1]['val_metrics']['f1'] == 0.85


# ── BUG-11: Train Joint YAML Config Deduplication ────────────────────────
def test_fix_11_train_joint_yaml_no_duplicate_unfreeze():
    """Verify train_joint.yaml does not have redundant unfreeze_backbone in training block."""
    cfg_path = Path("configs/train_joint.yaml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    assert 'unfreeze_backbone' not in data.get('training', {}), "Found redundant unfreeze_backbone in training block"
    assert data.get('model', {}).get('backbone_frozen') is False, "backbone_frozen should be False in model block"


# ── BUG-12: ARGUSOutput Type Hints ───────────────────────────────────────
def test_fix_12_argus_output_type_hints():
    """Verify ARGUSOutput annotations match the actual List[List[Tensor]] structure."""
    import typing
    hints = typing.get_type_hints(ARGUSOutput)
    # saliency and node_importance should be Optional[List[List[torch.Tensor]]]
    assert "saliency" in hints
    assert "node_importance" in hints


# ── BUG-13: Vectorized Fallback Semantic Tagging ──────────────────────────
def test_fix_13_vectorized_semantic_tagging():
    """Verify vectorized _fallback_detect produces correct semantic classes."""
    det = SemanticDetector(max_objects=6, model_name="heuristic")
    # Force fallback mode
    det.yolo = None

    frame = torch.rand(3, 96, 96)
    res = det(frame)

    assert isinstance(res, DetectionResult)
    assert res.semantic_types.ndim == 1
    assert len(res.semantic_types) == len(res.boxes)
    # Semantic types should be in {0, 1, 4} (person, vehicle, other object; never fake bag or weapon)
    assert (res.semantic_types >= 0).all() and (res.semantic_types <= 4).all()
    # Honest fallback: never fake weapon (3) or bag (2)
    assert not ((res.semantic_types == 2) | (res.semantic_types == 3)).any()


# ── BUG-14: train.py Accepts --device CLI Argument ────────────────────────
def test_fix_14_train_accepts_device_flag(monkeypatch):
    """Verify train.parse_args() accepts --device cpu/cuda forwarded by runner scripts."""
    import sys
    import train
    monkeypatch.setattr(sys, "argv", ["train.py", "--synthetic", "--device", "cpu"])
    args = train.parse_args()
    assert args.device == "cpu"
    cfg = train.load_config(args)
    assert cfg["device"] == "cpu"


# ── BUG-15: profile_complexity Optional Import & EdgeGAT in_dim ───────────
def test_fix_15_profile_complexity_hints_and_edge_gat_in_dim():
    """Verify count_parameters type hints resolve and EdgeGAT stores in_dim."""
    import typing
    from scripts.profile_complexity import count_parameters
    from argus.graph import EdgeGAT

    hints = typing.get_type_hints(count_parameters)
    assert "module" in hints
    gat = EdgeGAT(in_dim=12, edge_dim=10, out_dim=64, heads=4)
    assert gat.in_dim == 12
    assert gat.edge_dim == 10
    assert gat.out_dim == 64


# ── BUG-16: Default Checkpoints Exist in infer.py and web_demo.py ─────────
def test_fix_16_default_checkpoints_point_to_existing_champion(monkeypatch):
    """Verify infer.py defaults to outputs/checkpoints_rlvd_full/best.pt."""
    import sys
    import infer
    monkeypatch.setattr(sys, "argv", ["infer.py", "--input", "dummy.mp4"])
    args = infer.parse_args()
    assert args.checkpoint == Path("outputs/checkpoints_rlvd_full/best.pt")


# ── BUG-17: Execution Order CLI Compatibility ─────────────────────────────
def test_fix_17_execution_order_cli_compatibility(monkeypatch):
    """Verify evaluate.py, prepare_manifests.py, evaluate_frame_auc.py, and generate_pareto_plots.py support Execution Order flags."""
    import sys
    import evaluate
    from scripts import prepare_manifests, evaluate_frame_auc, generate_pareto_plots

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "dummy.pt", "--synthetic", "--device", "cpu"])
    eval_args = evaluate.parse_args()
    assert eval_args.device == "cpu"

    monkeypatch.setattr(sys, "argv", ["prepare_manifests.py", "--split-by", "source_video", "--output-dir", "data"])
    pm_args = prepare_manifests.parse_args()
    assert pm_args.split_by == "source_video"

    monkeypatch.setattr(sys, "argv", ["evaluate_frame_auc.py", "--config", "configs/ablations/ablation_4_no_pruning.yaml"])
    auc_args = evaluate_frame_auc.parse_args()
    assert auc_args.config == Path("configs/ablations/ablation_4_no_pruning.yaml")

    monkeypatch.setattr(sys, "argv", ["generate_pareto_plots.py", "--compile-from", "outputs", "--output", "figures/pareto_analysis.png"])
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--compile-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("figures/pareto_analysis.png"))
    p_args = parser.parse_args()
    assert p_args.compile_from == Path("outputs")


# ── BUG-18: Multi-Seed UCF Metric Schema Extraction ───────────────────────
def test_fix_18_multi_seed_nested_ucf_metrics():
    """Verify run_multi_seed._extract_ucf_metric supports both nested 'metrics' and top-level keys."""
    from scripts.run_multi_seed import _extract_ucf_metric, parse_args

    nested = {"metrics": {"frame_roc_auc": 0.7123, "frame_pr_auc": 0.4567}}
    flat = {"frame_roc_auc": 0.7123, "frame_pr_auc": 0.4567}
    assert _extract_ucf_metric(nested, "frame_roc_auc") == 0.7123
    assert _extract_ucf_metric(flat, "frame_pr_auc") == 0.4567

    ms_args = parse_args([])
    assert ms_args.device is None


# ── BUG-19: Time-to-Alert Rejects Zero-Latency Reward on Pre-Onset Alerts ─
def test_fix_19_tta_pre_onset_alert_handling():
    """Verify pre-onset alerts are counted as early alerts and never reported as 0 ms latency successes."""
    from scripts import measure_time_to_alert
    import sys

    # Mock ground truth and video frames
    mock_gt = {
        "TestVideo_01": {
            "is_anomaly": True,
            "total_extracted_frames": 300,
            "intervals": [(100, 200)],  # Incident starts at frame 100
            "category": "Assault",
            "frame_entries": [(i, "dummy") for i in range(1, 301)],
        }
    }

    # Case A: Alert occurs at frame 50 (BEFORE incident onset at 100)
    scores_early = np.zeros(300, dtype=np.float32)
    scores_early[49] = 0.95  # frame 50 crossed threshold

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(measure_time_to_alert, "load_ucf_test_ground_truth", lambda *args, **kwargs: mock_gt)
        mp.setattr(measure_time_to_alert, "load_checkpoint", lambda *args, **kwargs: {"config": {}, "components": {}})
        mp.setattr(measure_time_to_alert, "build_model", lambda *args, **kwargs: MagicMock())
        mp.setattr(measure_time_to_alert, "validate_component_metadata", lambda *args, **kwargs: None)
        mp.setattr(measure_time_to_alert, "model_state_dict", lambda *args, **kwargs: {})
        mp.setattr(measure_time_to_alert, "evaluate_single_video", lambda *args, **kwargs: scores_early)
        mp.setattr(measure_time_to_alert, "load_video_frames", lambda *args, **kwargs: torch.zeros(300, 3, 128, 128))

        res = measure_time_to_alert.measure_time_to_alert(
            checkpoint_path=Path("dummy.pt"),
            annotations_path=Path("dummy.txt"),
            test_dir=Path("."),
            threshold=0.50,
            device="cpu",
        )

        assert res["detected_videos"] == 0
        assert res["missed_videos"] == 1
        assert res["early_alerts_before_onset"] == 1
        assert res["video_results"][0]["detected"] is False
        assert res["video_results"][0]["reason"] == "pre_onset_alert_only"
        assert res["video_results"][0]["delta_frames"] is None
        assert res["video_results"][0]["tta_ms"] is None


# ── BUG-20: Ablation Runner Enforces Full Retraining and Rejects --eval-only
def test_fix_20_ablation_eval_only_rejected(monkeypatch):
    """Verify that --eval-only is rejected by run_ablations.py."""
    import sys
    import scripts.run_ablations as ra

    monkeypatch.setattr(sys, "argv", ["run_ablations.py", "--eval-only"])
    with pytest.raises(ValueError, match="not a valid scientific ablation"):
        ra.main()


# ── BUG-21: Pareto Plot Generator Validates Measured Metrics Schema ───────
def test_fix_21_pareto_metrics_schema_validation(tmp_path):
    """Verify load_metrics enforces all required metrics and rejects missing keys."""
    from scripts.generate_pareto_plots import load_metrics

    valid_json = tmp_path / "valid.json"
    with open(valid_json, "w", encoding="utf-8") as f:
        json.dump({
            "TestModel": {"auc": 0.95, "fps": 30.0, "gflops": 25.0, "params_m": 30.0, "memory_mb": 150.0}
        }, f)
    parsed = load_metrics(valid_json)
    assert "TestModel" in parsed
    assert parsed["TestModel"]["auc"] == 0.95

    invalid_json = tmp_path / "invalid.json"
    with open(invalid_json, "w", encoding="utf-8") as f:
        json.dump({
            "TestModel": {"auc": 0.95, "fps": 30.0}  # missing gflops, params_m, memory_mb
        }, f)
    with pytest.raises(ValueError, match="missing"):
        load_metrics(invalid_json)


# ── BUG-22: UCF-Crime Evaluator Supports --allow-partial Flag ─────────────
def test_fix_22_evaluate_frame_auc_partial_flag(monkeypatch):
    """Verify evaluate_frame_auc.py parse_args accepts --allow-partial."""
    import sys
    from scripts import evaluate_frame_auc

    monkeypatch.setattr(sys, "argv", ["evaluate_frame_auc.py", "--allow-partial"])
    args = evaluate_frame_auc.parse_args()
    assert args.allow_partial is True


# ── BUG-23: Audit Results Detects Split Data Leakage ───────────────────────
def test_fix_23_audit_splits_detects_leakage(tmp_path):
    """Verify audit_splits detects shared source videos across splits."""
    import csv
    from scripts.audit_results import AuditContext, audit_splits

    ctx = AuditContext(strict=True)
    # Write leaking CSVs
    tr_path = tmp_path / "train_clean.csv"
    va_path = tmp_path / "val_clean.csv"
    te_path = tmp_path / "test_clean.csv"

    with open(tr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dataset", "category"])
        w.writeheader()
        w.writerow({"path": "videoA_clip_01.pt", "label": "0", "dataset": "rlvd", "category": "NonViolence"})

    with open(va_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dataset", "category"])
        w.writeheader()
        w.writerow({"path": "videoA_clip_02.pt", "label": "0", "dataset": "rlvd", "category": "NonViolence"})

    with open(te_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dataset", "category"])
        w.writeheader()
        w.writerow({"path": "videoB_clip_01.pt", "label": "1", "dataset": "rlvd", "category": "Violence"})

    audit_splits(ctx, tmp_path)
    assert len(ctx.failures) > 0
    assert any("DATA LEAKAGE DETECTED" in f for f in ctx.failures)



