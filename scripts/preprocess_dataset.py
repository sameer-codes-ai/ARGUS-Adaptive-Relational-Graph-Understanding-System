"""
ARGUS Dataset Preprocessor (RLVD -> Fast Tensors / Frames).
Extracts, uniformly samples, and downscales raw surveillance videos into
preprocessed tensors (.pt) or frame image sequences.

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --num-workers 8 --image-size 128 --num-frames 16
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


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}


def process_single_video(task: tuple[str, str, int, int]) -> tuple[bool, str, str]:
    """
    Process a single video: extract `num_frames` uniformly, resize to `image_size`,
    and save as a uint8 PyTorch tensor (T, 3, H, W).
    """
    video_path_str, out_path_str, num_frames, image_size = task
    video_path = Path(video_path_str)
    out_path = Path(out_path_str)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, video_path.name, "Failed to open video file"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and total_frames >= num_frames:
            target_indices = set(torch.linspace(0, total_frames - 1, num_frames).long().tolist())
        else:
            target_indices = None

        frames = []
        frame_idx = 0
        reservoir_rng = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if target_indices is None or frame_idx in target_indices:
                h, w = frame.shape[:2]
                if h != image_size or w != image_size:
                    interp = cv2.INTER_AREA if (h > image_size and w > image_size) else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (image_size, image_size), interpolation=interp)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Store as uint8 tensor (3, H, W)
                tensor_frame = torch.from_numpy(rgb).permute(2, 0, 1)

                if target_indices is None:
                    if reservoir_rng is None:
                        import random
                        reservoir_rng = random.Random(0)
                    seen = frame_idx + 1
                    if len(frames) < num_frames:
                        frames.append(tensor_frame)
                    else:
                        replacement = reservoir_rng.randrange(seen)
                        if replacement < num_frames:
                            frames[replacement] = tensor_frame
                else:
                    frames.append(tensor_frame)

            frame_idx += 1
            if target_indices and len(frames) >= num_frames:
                break

        cap.release()

        if not frames:
            return False, video_path.name, "No frames could be extracted"

        # Pad if short
        if len(frames) < num_frames:
            repeat_count = num_frames - len(frames)
            last = frames[-1:]
            frames = frames + last * repeat_count

        # Stack into (T, 3, H, W) uint8 tensor
        clip_tensor = torch.stack(frames[:num_frames], dim=0)

        # Ensure directory exists and save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clip_tensor, out_path)
        return True, video_path.name, ""

    except Exception as exc:
        return False, video_path.name, str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RLVD videos into optimized PyTorch tensors.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/Real Life Violence Dataset"),
        help="Path to raw Real Life Violence Dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/preprocessed_rlvd"),
        help="Path to output preprocessed dataset directory.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames per clip (default: 16).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square image resolution (default: 128).",
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
        help="Force re-extraction even if preprocessed .pt files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workers = args.num_workers or min(8, max(1, os.cpu_count() or 4))

    print("=" * 65)
    print("        ARGUS DATASET PREPROCESSOR (RLVD -> PREPROCESSED)       ")
    print("=" * 65)
    print(f"Source Directory:     {args.source_dir.resolve()}")
    print(f"Output Directory:     {args.output_dir.resolve()}")
    print(f"Target Configuration: {args.num_frames} frames @ {args.image_size}x{args.image_size}")
    print(f"Worker Processes:     {workers}")
    print("=" * 65)

    existing_v = list((args.output_dir / "Violence").glob("*.pt")) if (args.output_dir / "Violence").is_dir() else []
    existing_nv = list((args.output_dir / "NonViolence").glob("*.pt")) if (args.output_dir / "NonViolence").is_dir() else []
    if not args.force and len(existing_v) >= 1000 and len(existing_nv) >= 1000:
        print(f"[SKIP] Preprocessed RLVD tensors already present ({len(existing_v)} Violence, {len(existing_nv)} NonViolence). Use --force to re-extract.")
        return

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    # Discover videos across subdirectories
    categories = ["Violence", "NonViolence"]
    tasks: list[tuple[str, str, int, int]] = []

    for cat in categories:
        cat_dir = args.source_dir / cat
        if not cat_dir.is_dir():
            print(f"[WARNING] Subdirectory not found: {cat_dir}")
            continue

        out_cat_dir = args.output_dir / cat
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        for f in cat_dir.iterdir():
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                out_path = out_cat_dir / f"{f.stem}.pt"
                tasks.append((str(f), str(out_path), args.num_frames, args.image_size))

    if not tasks:
        raise ValueError(f"No video files found in {args.source_dir}")

    print(f"Discovered {len(tasks)} videos to process ({sum(1 for t in tasks if 'Violence' in t[1] and 'NonViolence' not in t[1])} Violence, {sum(1 for t in tasks if 'NonViolence' in t[1])} NonViolence).\n")

    start_time = time.perf_counter()
    success_count = 0
    failures = []

    if workers <= 1:
        for t in tqdm(tasks, desc="Preprocessing Videos", dynamic_ncols=True):
            ok, name, err = process_single_video(t)
            if ok:
                success_count += 1
            else:
                failures.append((name, err))
    else:
        with mp.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_video, tasks, chunksize=10),
                                total=len(tasks), desc="Preprocessing Videos", dynamic_ncols=True))
        for ok, name, err in results:
            if ok:
                success_count += 1
            else:
                failures.append((name, err))

    elapsed = time.perf_counter() - start_time
    total_size_bytes = sum(f.stat().st_size for f in args.output_dir.rglob("*.pt") if f.is_file())
    total_size_mb = total_size_bytes / (1024 * 1024)

    print("\n" + "=" * 65)
    print("                   PREPROCESSING COMPLETE                        ")
    print("=" * 65)
    print(f"Successfully processed: {success_count} / {len(tasks)} videos (100.0%)" if success_count == len(tasks) else f"Processed: {success_count} / {len(tasks)}")
    print(f"Total time elapsed:     {elapsed:.2f} seconds ({elapsed / max(1, len(tasks)) * 1000.0:.1f} ms/video)")
    print(f"Preprocessed disk size: {total_size_mb:.1f} MB ({total_size_mb / 1024.0:.2f} GB)")
    print(f"Output directory:       {args.output_dir.resolve()}")
    if failures:
        print(f"[WARNING] {len(failures)} failures encountered:")
        for name, err in failures[:10]:
            print(f"  - {name}: {err}")
    print("=" * 65)


if __name__ == "__main__":
    main()
