"""
ARGUS UCF-Crime Test Preprocessor (Extracted Frames -> Atomic Video Tensors).
Packs all frames of each individual UCF-Crime test video into a single,
self-contained PyTorch tensor (.pt), preserving exact frame indices, temporal ordering,
and category metadata.

Usage:
    python scripts/preprocess_ucf_test.py
    python scripts/preprocess_ucf_test.py --num-workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
from tqdm import tqdm


def process_single_video_task(task: tuple[str, str, str, list[tuple[int, str]]]) -> tuple[bool, str, int, str]:
    """
    Load all frame PNGs for an individual test video, stack into a single uint8 tensor (N, 3, H, W),
    and save atomically with metadata.
    """
    stem, category, out_path_str, frame_entries = task
    out_path = Path(out_path_str)

    try:
        # Sort by frame index to guarantee exact temporal order
        sorted_entries = sorted(frame_entries, key=lambda x: x[0])
        frame_indices = [idx for idx, _ in sorted_entries]

        frames = []
        for _, p_str in sorted_entries:
            img = cv2.imread(p_str)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Store as uint8 tensor (3, H, W)
            tensor_frame = torch.from_numpy(rgb).permute(2, 0, 1)
            frames.append(tensor_frame)

        if not frames:
            return False, stem, 0, "No valid frames could be read"

        # Stack into (N, 3, H, W) uint8 tensor
        video_tensor = torch.stack(frames, dim=0)

        payload = {
            "frames": video_tensor,
            "frame_indices": frame_indices,
            "stem": stem,
            "category": category,
            "num_frames": len(frames),
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return True, stem, len(frames), ""

    except Exception as exc:
        return False, stem, 0, str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess UCF-Crime test frames into atomic video tensors.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/UCF-Crime/Test"),
        help="Path to extracted UCF-Crime test directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/preprocessed_ucf_test"),
        help="Path to output preprocessed UCF-Crime test directory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(cpu_count, 8)).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-packaging even if preprocessed UCF test tensors already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workers = args.num_workers or min(8, max(1, os.cpu_count() or 4))

    print("=" * 65)
    print("      ARGUS UCF-CRIME TEST PREPROCESSOR (PNGs -> .PT TENSORS)     ")
    print("=" * 65)
    print(f"Source Directory: {args.source_dir.resolve()}")
    print(f"Output Directory: {args.output_dir.resolve()}")
    print(f"Worker Processes: {workers}")
    print("=" * 65)

    idx_path = args.output_dir / "ucf_test_index.json"
    existing_pts = list(args.output_dir.glob("*/*.pt")) if args.output_dir.is_dir() else []
    if not args.force and idx_path.is_file() and len(existing_pts) >= 290:
        print(f"[SKIP] Preprocessed UCF-Crime test tensors already present ({len(existing_pts)} video .pt files + index). Use --force to re-package.")
        return

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    # Index all frames by video stem across all category folders
    videos_map: dict[str, dict] = {}
    total_pngs = 0

    print("Indexing test frames across category folders...")
    for cat_dir in sorted(args.source_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        for png_file in cat_dir.glob("*.png"):
            stem, _, suffix = png_file.stem.rpartition("_")
            try:
                f_idx = int(suffix)
                if stem not in videos_map:
                    videos_map[stem] = {
                        "category": cat_name,
                        "entries": [],
                    }
                videos_map[stem]["entries"].append((f_idx, str(png_file)))
                total_pngs += 1
            except ValueError:
                continue

    print(f"Found {len(videos_map)} individual test videos ({total_pngs} total PNG frames).\n")

    tasks = []
    for stem, data in videos_map.items():
        cat_name = data["category"]
        out_path = args.output_dir / cat_name / f"{stem}.pt"
        tasks.append((stem, cat_name, str(out_path), data["entries"]))

    start_time = time.perf_counter()
    success_count = 0
    total_frames_saved = 0
    failures = []

    if workers <= 1:
        for t in tqdm(tasks, desc="Packaging Videos", dynamic_ncols=True):
            ok, name, n_f, err = process_single_video_task(t)
            if ok:
                success_count += 1
                total_frames_saved += n_f
            else:
                failures.append((name, err))
    else:
        with mp.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_video_task, tasks, chunksize=5),
                                total=len(tasks), desc="Packaging Videos", dynamic_ncols=True))
        for ok, name, n_f, err in results:
            if ok:
                success_count += 1
                total_frames_saved += n_f
            else:
                failures.append((name, err))

    elapsed = time.perf_counter() - start_time
    total_size_bytes = sum(f.stat().st_size for f in args.output_dir.rglob("*.pt") if f.is_file())
    total_size_mb = total_size_bytes / (1024 * 1024)

    print("\n" + "=" * 65)
    print("                UCF-CRIME PREPROCESSING COMPLETE                 ")
    print("=" * 65)
    print(f"Successfully packaged: {success_count} / {len(tasks)} videos ({success_count / max(1, len(tasks)) * 100:.1f}%)")
    print(f"Total frames saved:    {total_frames_saved:,} frames")
    print(f"Total time elapsed:    {elapsed:.2f} seconds ({elapsed / max(1, len(tasks)) * 1000.0:.1f} ms/video)")
    print(f"Preprocessed size:     {total_size_mb:.1f} MB ({total_size_mb / 1024.0:.2f} GB)")
    print(f"Output directory:      {args.output_dir.resolve()}")
    if failures:
        print(f"[WARNING] {len(failures)} failures encountered:")
        for name, err in failures[:10]:
            print(f"  - {name}: {err}")
    print("=" * 65)


if __name__ == "__main__":
    main()
