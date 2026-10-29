"""
Quantitative Time-to-Alert (TTA) Benchmark for ARGUS (Phase 4.3).

Evaluates the real-time detection responsiveness of ARGUS on untrimmed surveillance
videos with annotated physical anomaly onset frames (UCF-Crime temporal test split).

Computes:
  - Frame Alert Latency: Delta f = f_alert - f_start
  - Wall-clock Alert Latency at 30 FPS: TTA_ms = (Delta f / 30.0) * 1000.0
  - Mean, Median, Min, Max, and Percentile (p90) Alert Latency
  - False Alert Rate before physical incident onset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

import numpy as np
import torch
from tqdm import tqdm

from argus.config import ARGUSConfig
from argus.checkpoint import load_checkpoint, model_state_dict, validate_component_metadata
from argus.baselines import build_model
from scripts.download_ucf_annotations import load_ucf_test_ground_truth
from scripts.evaluate_frame_auc import load_video_frames, evaluate_single_video


def measure_time_to_alert(
    checkpoint_path: Path,
    annotations_path: Path,
    test_dir: Path,
    threshold: float = 0.50,
    clip_len: int = 16,
    stride: int = 16,
    device: str = "cpu",
    output_json: Path | None = None,
    max_videos: int | None = None,
) -> dict:
    device = torch.device(device)
    print("=" * 65)
    print("      ARGUS QUANTITATIVE TIME-TO-ALERT (TTA) BENCHMARK (PHASE 4.3)  ")
    print("=" * 65)
    print(f"Model Checkpoint:    {checkpoint_path}")
    print(f"Alert Threshold:     tau = {threshold:.2f}")
    print(f"Device:              {device.type.upper()}")

    # 1. Ingest Annotations & Filter Anomalous Videos
    gt_dict = load_ucf_test_ground_truth(annotations_path, test_frames_root=test_dir)
    anom_videos = [
        stem
        for stem, v in gt_dict.items()
        if v.get("is_anomaly", False) and (v.get("total_extracted_frames", 0) > 0 or "pt_path" in v) and len(v.get("intervals", [])) > 0
    ]
    if max_videos:
        anom_videos = anom_videos[:max_videos]

    print(f"[INFO] Evaluating TTA on {len(anom_videos)} anomalous test videos with annotated onset intervals.")

    # 2. Load Model
    state = load_checkpoint(checkpoint_path, map_location=device, allow_unsafe=True)
    saved_cfg = state.get("config", {})
    cfg = ARGUSConfig.from_dict(saved_cfg, strict=True) if saved_cfg else ARGUSConfig().validate()
    model_type = state.get("model_type") or state.get("components", {}).get("model_type", "argus")
    model = build_model(model_type=model_type, config=cfg).to(device)
    validate_component_metadata(state, model)
    model.load_state_dict(model_state_dict(state))
    model.eval()

    tta_frames_list = []
    tta_ms_list = []
    early_alerts = 0  # Alerted before incident start
    detected_count = 0
    missed_count = 0
    video_details = []

    for stem in tqdm(anom_videos, desc="Measuring TTA", dynamic_ncols=True):
        v_info = gt_dict[stem]
        intervals = v_info["intervals"]
        f_start_first = intervals[0][0]  # First incident onset frame (1-based)
        frame_entries = v_info.get("frame_entries", [])

        if "pt_path" in v_info and Path(v_info["pt_path"]).is_file():
            pt_data = torch.load(v_info["pt_path"], map_location="cpu", weights_only=True)
            frames_tensor = pt_data["frames"] if isinstance(pt_data, dict) else pt_data
            if frames_tensor.shape[-1] != cfg.image_size:
                import torch.nn.functional as F
                frames_tensor = F.interpolate(frames_tensor.float() / 255.0, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
            else:
                frames_tensor = frames_tensor.float() / 255.0
        else:
            frames_tensor = load_video_frames(frame_entries, image_size=cfg.image_size)
        if frames_tensor.shape[0] == 0:
            continue

        scores = evaluate_single_video(
            model=model,
            frames_tensor=frames_tensor,
            clip_len=clip_len,
            stride=stride,
            aggregation="mean",
            streaming=True,
            stream_id=stem,
            device=device,
        )

        # Find frames where score crosses threshold and map to original 30 FPS video frame coordinates
        alert_indices = np.where(scores >= threshold)[0]
        if len(alert_indices) > 0:
            raw_alert_frames = [
                int(frame_entries[int(idx)][0]) if int(idx) < len(frame_entries) else int(idx) + 1
                for idx in alert_indices
            ]
            if any(f < f_start_first for f in raw_alert_frames):
                early_alerts += 1

            post_onset_alerts = [f for f in raw_alert_frames if f >= f_start_first]
            if post_onset_alerts:
                f_alert = post_onset_alerts[0]
            else:
                f_alert = raw_alert_frames[0]

            delta_frames = f_alert - f_start_first
            tta_ms = round((delta_frames / 30.0) * 1000.0, 1)

            detected_count += 1

            # Physical reaction latency from incident onset
            reaction_frames = max(0, delta_frames)
            reaction_ms = max(0.0, tta_ms)
            tta_frames_list.append(reaction_frames)
            tta_ms_list.append(reaction_ms)

            video_details.append(
                {
                    "video": stem,
                    "category": v_info["category"],
                    "incident_start_frame": f_start_first,
                    "alert_frame": f_alert,
                    "delta_frames": delta_frames,
                    "tta_ms": tta_ms,
                    "detected": True,
                }
            )
        else:
            missed_count += 1
            video_details.append(
                {
                    "video": stem,
                    "category": v_info["category"],
                    "incident_start_frame": f_start_first,
                    "alert_frame": None,
                    "delta_frames": None,
                    "tta_ms": None,
                    "detected": False,
                }
            )

    # Compute Statistics
    mean_tta_ms = float(np.mean(tta_ms_list)) if tta_ms_list else float("nan")
    median_tta_ms = float(np.median(tta_ms_list)) if tta_ms_list else float("nan")
    p90_tta_ms = float(np.percentile(tta_ms_list, 90)) if tta_ms_list else float("nan")
    min_tta_ms = float(np.min(tta_ms_list)) if tta_ms_list else float("nan")
    max_tta_ms = float(np.max(tta_ms_list)) if tta_ms_list else float("nan")

    detection_rate = detected_count / max(len(anom_videos), 1)

    results = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "total_anomalous_videos": len(anom_videos),
        "detected_videos": detected_count,
        "missed_videos": missed_count,
        "detection_rate": round(detection_rate, 4),
        "early_alerts_before_onset": early_alerts,
        "mean_tta_ms": round(mean_tta_ms, 1),
        "median_tta_ms": round(median_tta_ms, 1),
        "p90_tta_ms": round(p90_tta_ms, 1),
        "min_tta_ms": round(min_tta_ms, 1),
        "max_tta_ms": round(max_tta_ms, 1),
        "mean_tta_frames": round(float(np.mean(tta_frames_list)), 1) if tta_frames_list else None,
        "median_tta_frames": round(float(np.median(tta_frames_list)), 1) if tta_frames_list else None,
        "video_results": video_details,
    }

    print("\n--- Quantitative Time-to-Alert (TTA) Results ---")
    print(f"Incident Detection Rate: {detected_count}/{len(anom_videos)} ({detection_rate*100:.1f}%)")
    print(f"Mean Time-to-Alert:      {mean_tta_ms:.1f} ms ({results['mean_tta_frames']} frames @ 30 FPS)")
    print(f"Median Time-to-Alert:    {median_tta_ms:.1f} ms ({results['median_tta_frames']} frames)")
    print(f"90th Percentile TTA:     {p90_tta_ms:.1f} ms")
    print(f"Sub-Second Alerts (<1s): {sum(1 for t in tta_ms_list if t <= 1000.0)} / {len(tta_ms_list)}")
    print("=" * 65)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] TTA benchmark results saved to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure ARGUS Time-to-Alert latency.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs" / "checkpoints_rlvd_full" / "best.pt")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data" / "ucf_crime_temporal_annotations.txt")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "preprocessed_ucf_test" if (PROJECT_ROOT / "datasets" / "preprocessed_ucf_test").is_dir() else PROJECT_ROOT / "datasets" / "UCF-Crime" / "Test",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--output",
        "--output-json",
        "-o",
        type=Path,
        dest="output",
        default=PROJECT_ROOT / "outputs" / "profiling" / "time_to_alert_benchmark.json",
    )
    args = parser.parse_args()

    measure_time_to_alert(
        checkpoint_path=args.checkpoint,
        annotations_path=args.annotations,
        test_dir=args.test_dir,
        threshold=args.threshold,
        device=args.device,
        output_json=args.output,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
