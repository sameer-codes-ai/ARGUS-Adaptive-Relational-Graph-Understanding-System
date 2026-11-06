"""
Script to download and parse official UCF-Crime temporal test annotations (Sultani et al., CVPR 2018).
Saves raw annotations to data/ucf_crime_temporal_annotations.txt and builds frame-level ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
import numpy as np

# Authoritative mirrors for official UCF-Crime testing annotations
ANNOTATION_URLS = [
    "https://huggingface.co/datasets/jinmang2/ucf_crime/raw/main/UCF_Crimes-Train-Test-Split/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    "https://raw.githubusercontent.com/vikiDSL/VADTree/master/dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    "https://raw.githubusercontent.com/huiyuiui/ReactVAU-code/main/raw_annotations/ucf_database_test_anno.txt",
]


def download_annotations(destination: Path, urls: list[str] = ANNOTATION_URLS) -> str:
    """Download the raw annotation file if not already present."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[INFO] Annotation file already exists: {destination}")
        with open(destination, "r", encoding="utf-8") as f:
            return f.read()

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"[INFO] Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "ARGUS-Dataset-Loader/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
                if len(content.strip().splitlines()) >= 280:
                    with open(destination, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"[SUCCESS] Downloaded and saved annotations to {destination} ({len(content)} bytes)")
                    return content
        except Exception as e:
            print(f"[WARN] Failed fetching from {url}: {e}")
            last_err = e

    raise RuntimeError(f"Could not download annotations from any mirror. Last error: {last_err}")


def parse_temporal_annotation_line(line: str) -> dict | None:
    """
    Parse a single line of Sultani et al. annotation format:
    Format: Video_Name Category Start_Frame_1 End_Frame_1 Start_Frame_2 End_Frame_2
    Example: Abuse028_x264.mp4 Abuse 165 240 -1 -1
    Example: Arson011_x264.mp4 Arson 150 420 680 1267
    Example: Normal_Videos_003_x264.mp4 Normal -1 -1 -1 -1
    """
    parts = line.strip().split()
    if not parts:
        return None

    video_filename = parts[0]
    stem = Path(video_filename).stem
    category = parts[1]

    # Parse anomaly intervals
    intervals = []
    if len(parts) >= 4:
        try:
            s1, e1 = int(parts[2]), int(parts[3])
            if s1 >= 0 and e1 >= 0 and e1 >= s1:
                intervals.append((s1, e1))
        except ValueError:
            pass
    if len(parts) >= 6:
        try:
            s2, e2 = int(parts[4]), int(parts[5])
            if s2 >= 0 and e2 >= 0 and e2 >= s2:
                intervals.append((s2, e2))
        except ValueError:
            pass

    return {
        "video_file": video_filename,
        "stem": stem,
        "category": category,
        "is_anomaly": len(intervals) > 0,
        "intervals": intervals,
    }


def load_ucf_test_ground_truth(
    annotation_path: Path,
    test_frames_root: Path | None = None,
    subsample_step: int = 10,
) -> dict[str, dict]:
    """
    Load ground truth for UCF-Crime test set.
    If test_frames_root is provided, associates with extracted PNG frames.
    For each video:
      - 'frames': list of sorted (frame_index, frame_path)
      - 'y_true': binary np.ndarray of shape (num_frames,)
      - 'intervals': list of (start, end)
      - 'category': category name
      - 'is_anomaly': bool
    """
    if not annotation_path.exists():
        download_annotations(annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    annotations = {}
    for line in lines:
        parsed = parse_temporal_annotation_line(line)
        if parsed:
            annotations[parsed["stem"]] = parsed

    # 1. Check for preprocessed UCF index or .pt directory first
    index_file = None
    preprocessed_dir = None
    candidates = []
    if test_frames_root is not None:
        candidates.append(test_frames_root)
        candidates.append(test_frames_root.parent / "preprocessed_ucf_test")
    candidates.append(Path("datasets/preprocessed_ucf_test"))

    for c in candidates:
        if (c / "ucf_test_index.json").is_file():
            index_file = c / "ucf_test_index.json"
            preprocessed_dir = c
            break
        elif c.is_dir() and any(c.glob("*/*.pt")):
            preprocessed_dir = c
            break

    if index_file is not None and index_file.is_file():
        with open(index_file, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        for stem, info in annotations.items():
            if stem in idx_data:
                meta = idx_data[stem]
                frame_indices = meta["frame_indices"]
                num_frames = meta["num_frames"]
                pt_p = Path(meta["pt_path"])
                info["pt_path"] = pt_p if pt_p.is_file() else (preprocessed_dir / meta["category"] / f"{stem}.pt")
                info["total_extracted_frames"] = num_frames
                info["frame_entries"] = [(idx, info["pt_path"]) for idx in frame_indices]
                y_true = np.zeros(num_frames, dtype=np.int32)
                if info["is_anomaly"]:
                    for i, f_idx in enumerate(frame_indices):
                        for start_f, end_f in info["intervals"]:
                            if start_f <= f_idx <= end_f:
                                y_true[i] = 1
                                break
                info["y_true"] = y_true
            else:
                info["total_extracted_frames"] = 0
                info["y_true"] = np.array([], dtype=np.int32)
                info["frame_entries"] = []
        return annotations

    if test_frames_root is not None and test_frames_root.exists():
        # Fast single-pass file index across all category folders
        files_by_stem: dict[str, list[tuple[int, Path]]] = {}
        for cat_dir in test_frames_root.iterdir():
            if not cat_dir.is_dir():
                continue
            for png_file in cat_dir.glob("*.png"):
                stem, _, suffix = png_file.stem.rpartition("_")
                try:
                    f_idx = int(suffix)
                    files_by_stem.setdefault(stem, []).append((f_idx, png_file))
                except ValueError:
                    continue

        for stem, info in annotations.items():
            frame_entries = files_by_stem.get(stem, [])
            frame_entries.sort(key=lambda x: x[0])
            info["frame_entries"] = frame_entries
            num_frames = len(frame_entries)

            if num_frames == 0:
                info["y_true"] = np.array([], dtype=np.int32)
                continue

            # Generate frame-level binary ground truth (Option A: Subsampled Grid)
            y_true = np.zeros(num_frames, dtype=np.int32)
            if info["is_anomaly"]:
                for i, (f_idx, _) in enumerate(frame_entries):
                    for start_f, end_f in info["intervals"]:
                        if start_f <= f_idx <= end_f:
                            y_true[i] = 1
                            break

            info["y_true"] = y_true
            info["total_extracted_frames"] = num_frames

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Download and process UCF-Crime test temporal annotations.")
    parser.add_argument(
        "--output-txt",
        "--output",
        "-o",
        dest="output_txt",
        type=Path,
        default=Path("data/ucf_crime_temporal_annotations.txt"),
        help="Path to save raw annotation text file.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("datasets/UCF-Crime/Test"),
        help="Path to extracted UCF-Crime test frames folder.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/ucf_crime_test_meta.json"),
        help="Path to save parsed metadata JSON.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("      UCF-CRIME TEMPORAL ANNOTATION INGESTION (PHASE 0.1)     ")
    print("=" * 60)

    download_annotations(args.output_txt)

    print(f"\n[INFO] Validating against extracted test frames in: {args.test_dir}")
    gt = load_ucf_test_ground_truth(args.output_txt, test_frames_root=args.test_dir)

    total_videos = len(gt)
    anom_videos = sum(1 for v in gt.values() if v["is_anomaly"])
    norm_videos = total_videos - anom_videos
    total_frames = sum(v.get("total_extracted_frames", 0) for v in gt.values())
    total_anom_frames = sum(int(np.sum(v.get("y_true", []))) for v in gt.values())

    print(f"\nGround Truth Summary:")
    print(f"  Total test videos:          {total_videos:5d}")
    print(f"    - Normal videos:          {norm_videos:5d}")
    print(f"    - Anomaly videos:         {anom_videos:5d}")
    print(f"  Total extracted PNG frames: {total_frames:5d}")
    print(f"  Total anomalous frames:     {total_anom_frames:5d} ({total_anom_frames / max(total_frames, 1) * 100:.2f}%)")
    print(f"  Total normal frames:        {total_frames - total_anom_frames:5d}")

    # Save summary metadata JSON for quick loading
    meta_to_save = {}
    for stem, v in gt.items():
        meta_to_save[stem] = {
            "video_file": v["video_file"],
            "category": v["category"],
            "is_anomaly": v["is_anomaly"],
            "intervals": v["intervals"],
            "total_frames": v.get("total_extracted_frames", 0),
            "anom_frames": int(np.sum(v.get("y_true", []))),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(meta_to_save, f, indent=2)
    print(f"[SUCCESS] Saved metadata index to {args.output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
