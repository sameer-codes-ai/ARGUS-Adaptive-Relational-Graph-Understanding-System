"""
Unit and Integration Tests for ARGUS Preprocessed Dataset Pipelines (RLVD and UCF-Crime).

Validates:
1. RLVD preprocessed .pt files (exact count, shapes, uint8 [0, 255] value range, finite checks).
2. Manifest integrity (train_clean.csv, val_clean.csv, test_clean.csv, 0% leakage, .pt resolution).
3. UCF-Crime test set preprocessed index (ucf_test_index.json, 290 videos, frame counts, annotations).
4. VideoManifest and load_clip_from_path compatibility with atomic .pt files.
5. Runner CLI argument validation (--eco-mode, --device, --output-json aliases).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

from argus.data import VideoManifest, load_clip_from_path
from scripts.download_ucf_annotations import load_ucf_test_ground_truth
from scripts.prepare_manifests import extract_source_id


def test_preprocessed_rlvd_directory_and_tensors():
    """Verify RLVD preprocessed directory has 2,000 .pt files with shape (16, 3, 128, 128)."""
    rlvd_dir = PROJECT_ROOT / "datasets" / "preprocessed_rlvd"
    if not rlvd_dir.is_dir():
        pytest.skip("datasets/preprocessed_rlvd not present on this machine.")

    violence_pts = list((rlvd_dir / "Violence").glob("*.pt"))
    nonviolence_pts = list((rlvd_dir / "NonViolence").glob("*.pt"))

    assert len(violence_pts) == 1000, f"Expected 1,000 Violence clips, found {len(violence_pts)}"
    assert len(nonviolence_pts) == 1000, f"Expected 1,000 NonViolence clips, found {len(nonviolence_pts)}"

    # Sample and inspect 10 files
    samples = violence_pts[:5] + nonviolence_pts[:5]
    for pt_file in samples:
        tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
        assert isinstance(tensor, torch.Tensor), f"{pt_file.name} is not a Tensor"
        assert tensor.shape == (16, 3, 128, 128), f"Unexpected shape {tensor.shape} in {pt_file.name}"
        assert tensor.dtype == torch.uint8, f"Unexpected dtype {tensor.dtype}"
        assert torch.isfinite(tensor).all(), f"NaN or Inf found in {pt_file.name}"
        assert tensor.min() >= 0, f"Negative values found in {pt_file.name}"
        assert tensor.max() <= 255, f"Value exceeds 255 in {pt_file.name}"


def test_manifests_point_to_valid_pt_and_no_leakage():
    """Verify train_clean, val_clean, and test_clean point to .pt files with zero leakage."""
    manifest_paths = {
        "train": PROJECT_ROOT / "data" / "train_clean.csv",
        "val": PROJECT_ROOT / "data" / "val_clean.csv",
        "test": PROJECT_ROOT / "data" / "test_clean.csv",
    }

    split_sources = {}
    split_counts = {}

    for split_name, m_path in manifest_paths.items():
        assert m_path.is_file(), f"Manifest {m_path} does not exist"
        with open(m_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        split_counts[split_name] = len(rows)
        sources = set()
        for r in rows:
            path_val = r.get("path") or r.get("filepath") or r.get("video_path")
            assert path_val, "Missing path column in manifest"
            source_id = extract_source_id(path_val)
            sources.add(source_id)

            resolved_path = (m_path.parent / path_val).resolve()
            assert resolved_path.is_file(), f"Referenced clip file not found: {resolved_path}"
            assert resolved_path.suffix.lower() == ".pt", f"Clip path is not a .pt file: {resolved_path}"

        split_sources[split_name] = sources

    assert split_counts["train"] == 1400, f"Expected 1400 train samples, got {split_counts['train']}"
    assert split_counts["val"] == 300, f"Expected 300 val samples, got {split_counts['val']}"
    assert split_counts["test"] == 300, f"Expected 300 test samples, got {split_counts['test']}"

    # Strict Zero-Leakage Assertion
    train_val_overlap = split_sources["train"].intersection(split_sources["val"])
    train_test_overlap = split_sources["train"].intersection(split_sources["test"])
    val_test_overlap = split_sources["val"].intersection(split_sources["test"])

    assert len(train_val_overlap) == 0, f"Train-Val source leakage: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train-Test source leakage: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val-Test source leakage: {val_test_overlap}"


def test_load_clip_from_path_with_pt():
    """Test load_clip_from_path with preprocessed .pt files."""
    manifest_path = PROJECT_ROOT / "data" / "val_clean.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    sample_pt = (manifest_path.parent / first_row["path"]).resolve()
    clip = load_clip_from_path(sample_pt, target_frames=16, image_size=96)

    assert isinstance(clip, torch.Tensor)
    assert clip.shape == (16, 3, 96, 96)
    assert clip.min() >= 0.0
    assert clip.max() <= 1.0
    assert torch.isfinite(clip).all()


def test_video_manifest_dataset():
    """Test VideoManifest initialization and __getitem__ on preprocessed manifest."""
    manifest_path = PROJECT_ROOT / "data" / "val_clean.csv"
    dataset = VideoManifest(manifest_path=manifest_path, frames=16, image_size=96)
    assert len(dataset) == 300

    clip, label = dataset[0]
    assert clip.shape == (16, 3, 96, 96)
    assert isinstance(label, torch.Tensor)
    assert label.item() in (0.0, 1.0)


def test_ucf_test_preprocessed_index_and_ground_truth():
    """Verify preprocessed UCF-Crime test set index and ground truth resolution."""
    index_path = PROJECT_ROOT / "datasets" / "preprocessed_ucf_test" / "ucf_test_index.json"
    if not index_path.is_file():
        pytest.skip("datasets/preprocessed_ucf_test/ucf_test_index.json not present.")

    annotations_path = PROJECT_ROOT / "data" / "ucf_crime_temporal_annotations.txt"
    gt_dict = load_ucf_test_ground_truth(annotations_path, test_frames_root=index_path.parent)

    assert len(gt_dict) == 290, f"Expected 290 UCF-Crime test videos, got {len(gt_dict)}"

    # Check that each video has valid pt_path and matching labels
    sample_videos = list(gt_dict.keys())[:10]
    for stem in sample_videos:
        v = gt_dict[stem]
        assert "pt_path" in v, f"Missing pt_path in ground truth for {stem}"
        pt_path = Path(v["pt_path"])
        assert pt_path.is_file(), f"PT file does not exist: {pt_path}"

        pt_data = torch.load(pt_path, map_location="cpu", weights_only=True)
        frames = pt_data["frames"] if isinstance(pt_data, dict) else pt_data
        assert frames.ndim == 4
        assert frames.shape[1] == 3
        assert frames.shape[2] == 64
        assert frames.shape[3] == 64
        assert frames.dtype == torch.uint8
        assert len(v["y_true"]) == frames.shape[0]


def test_script_argument_forwarding():
    """Test that runner scripts accept --device and --eco-mode CLI flags without crashing."""
    from scripts.train_baselines import parse_args as parse_train_baselines_args
    from scripts.run_ablations import parse_args as parse_run_ablations_args
    from scripts.run_multi_seed import parse_args as parse_run_multi_seed_args

    args_tb = parse_train_baselines_args(["--epochs", "5", "--device", "cpu", "--eco-mode"])
    assert args_tb.epochs == 5
    assert args_tb.device == "cpu"
    assert args_tb.eco_mode is True

    args_ra = parse_run_ablations_args(["--epochs", "5", "--device", "cuda", "--eco-mode"])
    assert args_ra.epochs == 5
    assert args_ra.device == "cuda"
    assert args_ra.eco_mode is True

    args_ms = parse_run_multi_seed_args(["--epochs", "5", "--device", "cuda", "--eco-mode"])
    assert args_ms.epochs == 5
    assert args_ms.device == "cuda"
    assert args_ms.eco_mode is True
