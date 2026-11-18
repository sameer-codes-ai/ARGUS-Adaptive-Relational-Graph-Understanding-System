"""
Frame-Level Sliding-Window Evaluator for ARGUS on UCF-Crime (CVPR 2018 Official Benchmark).
Implements Phase 0.2 of the ARGUS Methodological Overhaul.

Computes:
  - Official Frame-Level ROC-AUC
  - Frame-Level PR-AUC (Average Precision)
  - Equal Error Rate (EER)
  - False Alarm Rate (FAR) on normal test videos at tau = 0.50
  - Frame-Level Confusion Matrix, F1, Accuracy, Precision, Recall
  - Per-Category Performance Breakdown
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure project root and .testdeps are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

import cv2
import numpy as np
import torch
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
except ImportError:
    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(np.int64).flatten()
        y_score = np.asarray(y_score).astype(np.float64).flatten()
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sorted_indices = np.argsort(-y_score)
        y_sorted = y_true[sorted_indices]
        scores_sorted = y_score[sorted_indices]
        distinct_ends = np.r_[np.flatnonzero(np.diff(scores_sorted)) + 1, len(scores_sorted)]
        tp = np.cumsum(y_sorted)[distinct_ends - 1]
        fp = np.cumsum(1 - y_sorted)[distinct_ends - 1]
        tpr = np.r_[0.0, tp / n_pos]
        fpr = np.r_[0.0, fp / n_neg]
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(trapz_fn(tpr, fpr))

    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(np.int64).flatten()
        y_score = np.asarray(y_score).astype(np.float64).flatten()
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sorted_indices = np.argsort(-y_score)
        y_sorted = y_true[sorted_indices]
        scores_sorted = y_score[sorted_indices]
        distinct_ends = np.r_[np.flatnonzero(np.diff(scores_sorted)) + 1, len(scores_sorted)]
        tp = np.cumsum(y_sorted)[distinct_ends - 1]
        fp = np.cumsum(1 - y_sorted)[distinct_ends - 1]
        precisions = tp / np.maximum(tp + fp, 1)
        recalls = tp / n_pos
        recall_steps = np.diff(np.r_[0.0, recalls])
        return float(np.sum(precisions * recall_steps))

    def roc_curve(y_true, y_score):
        y_true = np.asarray(y_true).astype(np.int64).flatten()
        y_score = np.asarray(y_score).astype(np.float64).flatten()
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        sorted_indices = np.argsort(-y_score)
        y_sorted = y_true[sorted_indices]
        scores_sorted = y_score[sorted_indices]
        distinct_ends = np.r_[np.flatnonzero(np.diff(scores_sorted)) + 1, len(scores_sorted)]
        tp = np.cumsum(y_sorted)[distinct_ends - 1]
        fp = np.cumsum(1 - y_sorted)[distinct_ends - 1]
        tpr = np.r_[0.0, tp / n_pos]
        fpr = np.r_[0.0, fp / n_neg]
        thresholds = np.r_[scores_sorted[0] + 1e-5, scores_sorted[distinct_ends - 1]]
        return fpr, tpr, thresholds
from tqdm import tqdm

from argus import ARGUS
from argus.baselines import build_model
from argus.config import ARGUSConfig
from argus.checkpoint import load_checkpoint, model_state_dict, validate_component_metadata
from scripts.download_ucf_annotations import load_ucf_test_ground_truth, download_annotations


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding threshold."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, threshold


def load_video_frames(frame_entries: list[tuple[int, Path]], image_size: int) -> torch.Tensor:
    """
    Load and resize all extracted PNG frames for a video in a single fast pass.
    Returns tensor of shape (N, 3, image_size, image_size) in [0.0, 1.0].
    """
    frames_list = []
    for _, f_path in frame_entries:
        img = cv2.imread(str(f_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h != image_size or w != image_size:
            interp = cv2.INTER_AREA if h > image_size else cv2.INTER_LINEAR
            img = cv2.resize(img, (image_size, image_size), interpolation=interp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_list.append(img)

    if not frames_list:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)

    arr = np.stack(frames_list, axis=0)  # (N, H, W, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def evaluate_single_video(
    model: ARGUS,
    frames_tensor: torch.Tensor,
    clip_len: int = 16,
    stride: int = 16,
    aggregation: str = "mean",
    streaming: bool = False,
    stream_id: str = "default",
    device: str = "cpu",
) -> np.ndarray:
    """
    Run sliding-window inference on a single untrimmed video.
    Returns:
      1D np.ndarray of shape (N,) containing frame-level anomaly scores in [0, 1].
    """
    num_frames = frames_tensor.shape[0]
    if num_frames == 0:
        return np.array([], dtype=np.float32)

    def _interp_clip_scores(raw_idx, raw_scores) -> np.ndarray:
        sel_idx = raw_idx.detach().cpu().numpy() if isinstance(raw_idx, torch.Tensor) else np.asarray(raw_idx)
        sel_scores = raw_scores.detach().cpu().numpy() if isinstance(raw_scores, torch.Tensor) else np.asarray(raw_scores)
        if len(sel_idx) == 0:
            return np.zeros(clip_len, dtype=np.float32)
        if len(sel_idx) == 1:
            return np.full(clip_len, sel_scores[0], dtype=np.float32)
        order = np.argsort(sel_idx, kind="mergesort")
        return np.interp(np.arange(clip_len), sel_idx[order], sel_scores[order])

    try:
        # Handle videos shorter than clip_len
        if num_frames < clip_len:
            pad_count = clip_len - num_frames
            pad_frames = frames_tensor[-1:].repeat(pad_count, 1, 1, 1)
            padded_clip = torch.cat([frames_tensor, pad_frames], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(padded_clip, stream=streaming, stream_id=stream_id)
                short_scores = _interp_clip_scores(out.sampled_indices[0], out.scores[0])
            return np.clip(short_scores[:num_frames], 0.0, 1.0)

        # Generate sliding window start indices
        window_starts = list(range(0, num_frames - clip_len + 1, stride))
        # Ensure the tail frames of the video are covered
        if window_starts[-1] + clip_len < num_frames:
            window_starts.append(num_frames - clip_len)

        accum_scores = np.zeros(num_frames, dtype=np.float32)
        accum_counts = np.zeros(num_frames, dtype=np.float32)
        if aggregation == "max":
            accum_scores.fill(-1e9)

        for start_idx in window_starts:
            clip = frames_tensor[start_idx : start_idx + clip_len].unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(clip, stream=streaming, stream_id=stream_id)
                w_scores = _interp_clip_scores(out.sampled_indices[0], out.scores[0])

            if aggregation == "mean":
                accum_scores[start_idx : start_idx + clip_len] += w_scores
                accum_counts[start_idx : start_idx + clip_len] += 1.0
            elif aggregation == "max":
                accum_scores[start_idx : start_idx + clip_len] = np.maximum(
                    accum_scores[start_idx : start_idx + clip_len], w_scores
                )
                accum_counts[start_idx : start_idx + clip_len] = 1.0
    finally:
        # Reset streaming FIFO memory after finishing this video stream
        if streaming and hasattr(model, "memory") and model.memory is not None:
            model.memory.reset(stream_id=stream_id)

    if aggregation == "mean":
        final_scores = accum_scores / np.maximum(accum_counts, 1.0)
    else:
        accum_scores[accum_scores < -1e8] = 0.0
        final_scores = accum_scores

    return np.clip(final_scores, 0.0, 1.0)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Official UCF-Crime Frame-Level AUC Evaluator for ARGUS.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints_rlvd_full/best.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/ucf_crime_temporal_annotations.txt"),
        help="Path to official temporal annotations.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("datasets/preprocessed_ucf_test") if Path("datasets/preprocessed_ucf_test").is_dir() else Path("datasets/UCF-Crime/Test"),
        help="Directory containing preprocessed video tensors (.pt) or extracted test frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Temporal sliding window stride S (e.g. 16 for non-overlapping, 8 for 50%% overlap).",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="Sliding window temporal clip length L.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Score aggregation method for overlapping frames.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run model with causal FIFO streaming memory enabled.",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/eval_ucf_frame_level.json"),
        help="Path to save evaluation metrics JSON.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on number of test videos for quick verification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cpu or cuda). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow evaluation on fewer than the complete 290 test videos (for smoke/debug tests only).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file to override model inference settings (e.g. for ablations).",
    )
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        default=False,
        help="Allow unsafe pickle checkpoint loading.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    print("=" * 65)
    print("      ARGUS OFFICIAL UCF-CRIME FRAME-LEVEL EVALUATION (PHASE 0)     ")
    print("=" * 65)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:       {device.upper()}")
    print(f"Model Checkpoint:     {args.checkpoint}")
    print(f"Annotation File:      {args.annotations}")
    print(f"Test Frames Directory:{args.test_dir}")
    print(f"Sliding Window:       Length L={args.clip_len}, Stride S={args.stride}")
    print(f"Score Aggregation:    {args.aggregation.upper()}")
    print(f"Streaming Memory:     {'ENABLED' if args.streaming else 'DISABLED'}")
    print("=" * 65)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 1. Ingest Annotations & Ground Truth
    gt_dict = load_ucf_test_ground_truth(args.annotations, test_frames_root=args.test_dir)
    valid_videos = [stem for stem, v in gt_dict.items() if v.get("total_extracted_frames", 0) > 0]
    if args.max_videos:
        half = max(1, args.max_videos // 2)
        anom_v = [v for v in valid_videos if gt_dict[v]["is_anomaly"]][:half]
        norm_v = [v for v in valid_videos if not gt_dict[v]["is_anomaly"]][: args.max_videos - len(anom_v)]
        valid_videos = anom_v + norm_v

    EXPECTED_UCF_TEST_VIDEOS = 290
    if args.max_videos is None and not args.allow_partial:
        if len(valid_videos) < EXPECTED_UCF_TEST_VIDEOS:
            raise ValueError(
                f"Incomplete UCF-Crime test set: found {len(valid_videos)} valid videos with frames/tensors, "
                f"expected {EXPECTED_UCF_TEST_VIDEOS}. For scientifically valid publication benchmarks, "
                f"all {EXPECTED_UCF_TEST_VIDEOS} test videos must be evaluated. To run a debug/smoke test, "
                "pass --allow-partial or --max-videos N."
            )

    print(f"[INFO] Loaded ground truth for {len(valid_videos)} test videos with extracted frames.")

    # 2. Load Model & Config
    state = load_checkpoint(args.checkpoint, map_location=device, allow_unsafe=args.allow_unsafe_checkpoint)
    saved_cfg = dict(state.get("config", {}) or {}) if isinstance(state, dict) else {}
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if isinstance(yaml_cfg.get("model"), dict):
            saved_cfg.update(yaml_cfg["model"])
    cfg = ARGUSConfig.from_dict(saved_cfg, strict=True) if saved_cfg else ARGUSConfig().validate()

    model_type = state.get("model_type") or (state.get("components") or {}).get("model_type", "argus")
    model = build_model(model_type=model_type, config=cfg).to(device)
    validate_component_metadata(state, model)
    model.load_state_dict(model_state_dict(state))
    model.eval()

    epoch = state.get("epoch", "?") if isinstance(state, dict) else "?"
    print(f"[INFO] Checkpoint successfully loaded (Trained Epoch {epoch})")

    # 3. Sliding-Window Inference across all videos
    all_scores: list[float] = []
    all_labels: list[int] = []
    normal_scores: list[float] = []

    # Category breakdown tracking
    category_data: dict[str, dict[str, list]] = {}

    total_frames_processed = 0
    start_eval_time = time.perf_counter()

    pbar = tqdm(valid_videos, desc="Evaluating Videos", unit="video", dynamic_ncols=True)
    for stem in pbar:
        v_info = gt_dict[stem]
        frame_entries = v_info["frame_entries"]
        y_true = v_info["y_true"]
        category = v_info["category"]
        is_anomaly = v_info["is_anomaly"]

        # Load frames tensor for this video (supports fast atomic .pt or loose PNGs)
        if "pt_path" in v_info and Path(v_info["pt_path"]).is_file():
            pt_data = torch.load(v_info["pt_path"], map_location="cpu", weights_only=True)
            frames_tensor = pt_data["frames"] if isinstance(pt_data, dict) else pt_data
            if not frames_tensor.is_floating_point():
                frames_tensor = frames_tensor.float() / 255.0
            else:
                frames_tensor = frames_tensor.float()
            if frames_tensor.shape[-2:] != (cfg.image_size, cfg.image_size):
                import torch.nn.functional as F
                frames_tensor = F.interpolate(frames_tensor, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
        else:
            frames_tensor = load_video_frames(frame_entries, image_size=cfg.image_size)
        if frames_tensor.shape[0] == 0:
            continue

        # Run sliding window evaluation
        video_scores = evaluate_single_video(
            model=model,
            frames_tensor=frames_tensor,
            clip_len=args.clip_len,
            stride=args.stride,
            aggregation=args.aggregation,
            streaming=args.streaming,
            stream_id=stem,
            device=device,
        )

        n_pts = min(len(video_scores), len(y_true))
        scores_valid = video_scores[:n_pts]
        labels_valid = y_true[:n_pts]

        all_scores.extend(scores_valid.tolist())
        all_labels.extend(labels_valid.tolist())
        total_frames_processed += n_pts

        if not is_anomaly:
            normal_scores.extend(scores_valid.tolist())

        # Track per category
        if category not in category_data:
            category_data[category] = {"scores": [], "labels": []}
        category_data[category]["scores"].extend(scores_valid.tolist())
        category_data[category]["labels"].extend(labels_valid.tolist())

        pbar.set_postfix({"Frames": total_frames_processed})

    pbar.close()
    elapsed_time = time.perf_counter() - start_eval_time

    # 4. Compute Official Frame-Level Metrics
    y_true_all = np.array(all_labels, dtype=np.int32)
    y_scores_all = np.array(all_scores, dtype=np.float32)

    has_both_classes = len(np.unique(y_true_all)) > 1
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true_all, y_scores_all))
        pr_auc = float(average_precision_score(y_true_all, y_scores_all))
        eer, eer_thresh = compute_eer(y_true_all, y_scores_all)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
        eer, eer_thresh = float("nan"), 0.5

    # False Alarm Rate (FAR) on normal test videos
    if normal_scores:
        norm_arr = np.array(normal_scores, dtype=np.float32)
        far_50 = float(np.mean(norm_arr >= 0.50))
        far_eer = float(np.mean(norm_arr >= eer_thresh))
    else:
        far_50 = float("nan")
        far_eer = float("nan")

    # Binary Classification Metrics at tau = 0.50
    preds_50 = (y_scores_all >= 0.50).astype(np.int32)
    tp = int(np.sum((preds_50 == 1) & (y_true_all == 1)))
    fp = int(np.sum((preds_50 == 1) & (y_true_all == 0)))
    fn = int(np.sum((preds_50 == 0) & (y_true_all == 1)))
    tn = int(np.sum((preds_50 == 0) & (y_true_all == 0)))

    total = max(len(y_true_all), 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)

    fps = total_frames_processed / max(elapsed_time, 1e-6)
    ms_per_frame = (elapsed_time / max(total_frames_processed, 1)) * 1000.0

    # Per-category metrics
    category_summary = {}
    for cat, data in sorted(category_data.items()):
        cat_y = np.array(data["labels"], dtype=np.int32)
        cat_s = np.array(data["scores"], dtype=np.float32)
        cat_auc = float(roc_auc_score(cat_y, cat_s)) if len(np.unique(cat_y)) > 1 else None
        category_summary[cat] = {
            "total_frames": len(cat_y),
            "anomaly_frames": int(np.sum(cat_y)),
            "mean_anomaly_score": round(float(np.mean(cat_s)), 4),
            "max_anomaly_score": round(float(np.max(cat_s)), 4),
            "roc_auc": round(cat_auc, 4) if cat_auc is not None else "N/A (single class)",
        }

    results = {
        "protocol": "UCF-Crime Official Frame-Level Sliding Window (Sultani et al. CVPR 2018)",
        "checkpoint": str(args.checkpoint),
        "sliding_window": {
            "clip_len": args.clip_len,
            "stride": args.stride,
            "aggregation": args.aggregation,
            "streaming": args.streaming,
        },
        "metrics": {
            "frame_roc_auc": round(roc_auc, 4),
            "frame_pr_auc": round(pr_auc, 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_thresh, 4),
            "false_alarm_rate_tau_050": round(far_50, 4),
            "false_alarm_rate_tau_eer": round(far_eer, 4),
            "accuracy_tau_050": round(acc, 4),
            "precision_tau_050": round(prec, 4),
            "recall_tau_050": round(rec, 4),
            "f1_score_tau_050": round(f1, 4),
        },
        "confusion_matrix_tau_050": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "performance": {
            "total_videos_evaluated": len(valid_videos),
            "expected_total_videos": EXPECTED_UCF_TEST_VIDEOS,
            "is_complete_benchmark": len(valid_videos) == EXPECTED_UCF_TEST_VIDEOS,
            "total_frames_evaluated": total_frames_processed,
            "elapsed_seconds": round(elapsed_time, 2),
            "effective_fps": round(fps, 2),
            "latency_ms_per_frame": round(ms_per_frame, 2),
        },
        "category_breakdown": category_summary,
    }

    # Print Official Results Summary
    print("\n" + "=" * 65)
    print("          OFFICIAL UCF-CRIME FRAME-LEVEL EVALUATION REPORT         ")
    print("=" * 65)
    print(f"  Total Videos Evaluated:    {len(valid_videos):d}")
    print(f"  Total Frames Evaluated:    {total_frames_processed:d}")
    print(f"  Evaluation Duration:       {format_duration(elapsed_time)} ({fps:.2f} FPS)")
    print("-" * 65)
    print(f"  Frame-Level ROC-AUC:       {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"  Frame-Level PR-AUC:        {pr_auc:.4f}")
    print(f"  Equal Error Rate (EER):    {eer:.4f} (at threshold {eer_thresh:.4f})")
    print(f"  False Alarm Rate (Normal): {far_50:.4f} ({far_50*100:.2f}% at tau=0.50)")
    print("-" * 65)
    print(f"  Precision:                 {prec:.4f}")
    print(f"  Recall:                    {rec:.4f}")
    print(f"  F1-Score:                  {f1:.4f}")
    print(f"  Accuracy:                  {acc*100:.2f}%")
    print(f"  Confusion Matrix:          TP={tp:d} | FP={fp:d} | TN={tn:d} | FN={fn:d}")
    print("=" * 65)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Full results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
