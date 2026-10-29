"""
Unit tests for Phase 1: Split Unit, Data Integrity & Checkpoint Selection.
Tests:
  - Source video ID extraction and grouping
  - Strict zero data leakage verification across Train, Val, and Test splits
  - Checkpoint selection protocol on Validation ROC-AUC with loss tie-breaker
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_manifests import extract_source_id, split_records, verify_no_data_leakage


def test_extract_source_id():
    assert extract_source_id("video123_clip_001.mp4") == "video123"
    assert extract_source_id("Abuse028_x264_10.png") == "Abuse028"
    assert extract_source_id("Normal_Videos_003_x264_20.png") == "Normal_Videos_003"
    assert extract_source_id("V_999.mp4") == "V_999"
    assert extract_source_id("NV_101.mp4") == "NV_101"
    assert extract_source_id(Path("datasets/RLVD/Violence/V_42.mp4")) == "V_42"


def test_split_records_no_leakage():
    # Construct multiple clips derived from the same source videos
    records = []
    for vid_id in range(100):
        # Each video has 3 clips
        for clip_id in range(3):
            records.append({
                "path": f"/data/videos/video_{vid_id:03d}_clip_{clip_id:02d}.mp4",
                "source_id": f"video_{vid_id:03d}",
                "label": vid_id % 2,
                "category": "Violence" if vid_id % 2 == 1 else "Normal",
                "dataset": "TestDataset"
            })

    train, val, test = split_records(records, train_r=0.70, val_r=0.15, test_r=0.15, seed=42)

    # All 300 clips must be accounted for
    assert len(train) + len(val) + len(test) == 300

    # Ensure zero data leakage
    verify_no_data_leakage(train, val, test)

    # Verify that clips from the same source video NEVER appear in different splits
    train_sources = {r["source_id"] for r in train}
    val_sources = {r["source_id"] for r in val}
    test_sources = {r["source_id"] for r in test}

    assert len(train_sources.intersection(val_sources)) == 0
    assert len(train_sources.intersection(test_sources)) == 0
    assert len(val_sources.intersection(test_sources)) == 0


def test_checkpoint_selection_logic():
    # Simulation of train.py checkpoint selection rules
    best_score = 0.9000
    best_loss = 0.2500

    def should_save_best(current_auc, current_loss, best_auc, best_loss_val):
        if current_auc > best_auc + 1e-4:
            return True
        elif abs(current_auc - best_auc) <= 1e-4 and current_loss < best_loss_val:
            return True
        return False

    # 1. Strict improvement saves
    assert should_save_best(0.9050, 0.2600, best_score, best_loss) is True

    # 2. Insignificant score difference with higher loss does NOT save
    assert should_save_best(0.90005, 0.2600, best_score, best_loss) is False

    # 3. Tie with lower loss DOES save (deterministic tie-breaker)
    assert should_save_best(0.90005, 0.2300, best_score, best_loss) is True

    # 4. Worse score does NOT save even with much lower loss
    assert should_save_best(0.8800, 0.1000, best_score, best_loss) is False
