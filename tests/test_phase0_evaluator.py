"""
Unit tests for Phase 0 components:
  - Sultani annotation parsing and frame ground truth mapping (download_ucf_annotations.py)
  - Sliding-window frame evaluation, EER, and FAR calculations (evaluate_frame_auc.py)
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

from scripts.download_ucf_annotations import parse_temporal_annotation_line
from scripts.evaluate_frame_auc import compute_eer, evaluate_single_video
from argus.config import ARGUSConfig
from argus.model import ARGUS


def test_parse_annotation_single_interval():
    line = "Abuse028_x264.mp4 Abuse 165 240 -1 -1"
    parsed = parse_temporal_annotation_line(line)
    assert parsed is not None
    assert parsed["stem"] == "Abuse028_x264"
    assert parsed["category"] == "Abuse"
    assert parsed["is_anomaly"] is True
    assert parsed["intervals"] == [(165, 240)]


def test_parse_annotation_multi_interval():
    line = "Arson011_x264.mp4 Arson 150 420 680 1267"
    parsed = parse_temporal_annotation_line(line)
    assert parsed is not None
    assert parsed["stem"] == "Arson011_x264"
    assert parsed["category"] == "Arson"
    assert parsed["is_anomaly"] is True
    assert parsed["intervals"] == [(150, 420), (680, 1267)]


def test_parse_annotation_normal_video():
    line = "Normal_Videos_003_x264.mp4 Normal -1 -1 -1 -1"
    parsed = parse_temporal_annotation_line(line)
    assert parsed is not None
    assert parsed["stem"] == "Normal_Videos_003_x264"
    assert parsed["category"] == "Normal"
    assert parsed["is_anomaly"] is False
    assert parsed["intervals"] == []


def test_compute_eer_synthetic():
    # Construct synthetic distributions
    # 50 normal samples at ~0.2, 50 anomaly samples at ~0.8
    y_true = np.array([0] * 50 + [1] * 50)
    y_scores = np.array([0.1 + 0.2 * (i / 50) for i in range(50)] + [0.7 + 0.2 * (i / 50) for i in range(50)])
    eer, thresh = compute_eer(y_true, y_scores)
    assert not np.isnan(eer)
    assert 0.0 <= eer <= 0.1  # Perfectly separable should give ~0 EER


def test_evaluate_single_video_synthetic():
    # Create lightweight mock or minimal ARGUS model
    cfg = ARGUSConfig(
        dim=32,
        heads=2,
        use_pretrained_backbone=False,
        use_semantic_detector=False,
        node_visual_dim=0,
        temporal_layers=1,
        image_size=32,
        patch_size=16,
    )
    model = ARGUS(cfg).eval()

    # Synthetic video with 35 frames (should create 3 windows of length 16 with stride 16)
    frames_tensor = torch.rand(35, 3, 32, 32)
    scores = evaluate_single_video(
        model=model,
        frames_tensor=frames_tensor,
        clip_len=16,
        stride=16,
        aggregation="mean",
        streaming=False,
        device="cpu",
    )
    assert len(scores) == 35
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
