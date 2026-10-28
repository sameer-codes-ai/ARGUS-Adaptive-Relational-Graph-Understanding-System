"""
Dataset manifest generator and linker for ARGUS.
Scans external or local surveillance datasets and generates structured CSV manifests for training, validation, and testing.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
import sys
import time
from tqdm import tqdm

DEFAULT_DATASET_ROOT = Path(r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets")
DEFAULT_PROCESSED_ROOT = Path(r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\processed\train")

ANOMALY_CATEGORIES = {
    "abuse", "arrest", "arson", "assault", "burglary", "explosion", "fighting",
    "roadaccidents", "robbery", "shooting", "shoplifting", "stealing", "vandalism",
    "violence", "fire", "theft", "accident", "violencep"
}

NORMAL_CATEGORIES = {
    "normal", "normalvideos", "nonviolence", "nonviolencep", "shanghaitech"
}

MEDIA_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.pt'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _media_entries(directory: Path) -> list[Path]:
    """Return supported video/tensor files and frame directories."""
    entries = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
            entries.append(item)
        elif item.is_dir():
            frame_root = item / 'frames' if (item / 'frames').is_dir() else item
            if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in frame_root.iterdir()):
                entries.append(item)
    return entries


def _portable_path(path: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _category_label(category: str) -> int:
    normalized = category.lower()
    if normalized in NORMAL_CATEGORIES:
        return 0
    if normalized in ANOMALY_CATEGORIES:
        return 1
    raise ValueError(f"Unknown dataset category {category!r}; add an explicit label mapping")


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 0:
        return "0s"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARGUS video manifests from datasets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Path to raw datasets folder.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="Path to processed clips folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory for CSV manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Proportion of samples for training.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion of samples for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Proportion of samples for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("rlvd", "ucf", "shanghai", "processed"),
        default=("rlvd", "ucf", "shanghai", "processed"),
        help="Dataset sources to scan. Use '--datasets rlvd' for an RLVD-only run.",
    )
    return parser.parse_args()


def discover_real_life_violence(root: Path) -> list[dict]:
    records = []
    base = next(
        (
            candidate
            for candidate in (
                root / "RealLifeViolenceDataset",
                root / "Real Life Violence Dataset",
            )
            if candidate.is_dir()
        ),
        None,
    )
    if base is None:
        return records

    categories = [
        (("ViolenceP", "Violence"), 1, "Violence"),
        (("NonViolenceP", "NonViolence"), 0, "Normal"),
    ]
    for candidate_dirs, label_val, cat_name in categories:
        target = next((base / name for name in candidate_dirs if (base / name).is_dir()), None)
        if target is None:
            continue
        media = _media_entries(target)
        for video_dir in tqdm(media, desc=f"Scanning RealLifeViolence ({cat_name})", leave=False, dynamic_ncols=True):
            records.append({
                "path": str(video_dir.resolve()),
                "label": label_val,
                "category": cat_name,
                "dataset": "RealLifeViolence"
            })
    return records


def discover_ucf_crime(root: Path) -> list[dict]:
    records = []
    organized = root / "organized" / "ucf" / "train"
    if organized.is_dir():
        cat_dirs = [d for d in organized.iterdir() if d.is_dir()]
        for cat_dir in tqdm(cat_dirs, desc="Scanning UCF-Crime (Organized)", leave=False, dynamic_ncols=True):
            cat_name = cat_dir.name
            label_val = _category_label(cat_name)
            for video_dir in _media_entries(cat_dir):
                if video_dir.is_dir() or video_dir.suffix.lower() in MEDIA_EXTENSIONS:
                    records.append({
                        "path": str(video_dir.resolve()),
                        "label": label_val,
                        "category": cat_name,
                        "dataset": "UCF-Crime"
                    })
        return records

    ucf_raw = root / "UCF-Crime" / "Train"
    if ucf_raw.is_dir():
        cat_dirs = [d for d in ucf_raw.iterdir() if d.is_dir()]
        for cat_dir in tqdm(cat_dirs, desc="Scanning UCF-Crime (Raw)", leave=False, dynamic_ncols=True):
            cat_name = cat_dir.name
            label_val = _category_label(cat_name)
            for media in _media_entries(cat_dir):
                records.append({
                    "path": str(media.resolve()),
                    "source_id": media.stem,
                    "label": label_val,
                    "category": cat_name,
                    "dataset": "UCF-Crime"
                })
    return records


def discover_shanghaitech(root: Path) -> list[dict]:
    records = []
    st_dir = root / "ShanghaiTech" / "dataset" / "mp"
    if not st_dir.is_dir():
        return records

    video_dirs = [d for d in st_dir.iterdir() if d.is_dir()]
    for video_dir in tqdm(video_dirs, desc="Scanning ShanghaiTech", leave=False, dynamic_ncols=True):
        records.append({
            "path": str(video_dir.resolve()),
            "label": 0,
            "category": "Normal",
            "dataset": "ShanghaiTech"
        })
    return records


def discover_processed_clips(root: Path) -> list[dict]:
    records = []
    if not root.is_dir():
        return records

    cat_dirs = [d for d in root.iterdir() if d.is_dir()]
    for cat_dir in tqdm(cat_dirs, desc="Scanning Processed Clips", leave=False, dynamic_ncols=True):
        cat_name = cat_dir.name
        label_val = _category_label(cat_name)
        for clip_dir in _media_entries(cat_dir):
            if clip_dir.is_dir() or clip_dir.suffix.lower() in MEDIA_EXTENSIONS:
                records.append({
                    "path": str(clip_dir.resolve()),
                    "label": label_val,
                    "category": cat_name,
                    "dataset": "ProcessedClips"
                })
    return records


def split_records(records: list[dict], train_r: float, val_r: float, test_r: float, seed: int):
    if min(train_r, val_r, test_r) < 0 or abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")
    random.seed(seed)
    buckets: dict[tuple, list[dict]] = {}
    for r in records:
        source_id = r.get("source_id") or Path(r["path"]).stem.split("_clip_")[0]
        key = (r["dataset"], r["category"], r["label"], source_id)
        buckets.setdefault(key, []).append(r)

    train, val, test = [], [], []
    strata: dict[tuple, list[list[dict]]] = {}
    for key, group in buckets.items():
        strata.setdefault(key[:3], []).append(group)
    for groups in strata.values():
        random.shuffle(groups)
        n = len(groups)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        for group in groups[:n_train]:
            train.extend(group)
        for group in groups[n_train:n_train + n_val]:
            val.extend(group)
        for group in groups[n_train + n_val:]:
            test.extend(group)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_csv(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "dataset", "category"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            row["path"] = _portable_path(Path(row["path"]), path.parent)
            writer.writerow(row)
    print(f"  [OK] Saved {len(records):5d} rows to {path}")


def main():
    args = parse_args()
    start_time = time.perf_counter()
    print("=" * 55)
    print("        ARGUS SURVEILLANCE DATASET MANIFEST SCAN        ")
    print("=" * 55)

    print("Scanning dataset sources...")
    selected = set(args.datasets)

    rl_records = discover_real_life_violence(args.dataset_root) if "rlvd" in selected else []
    print(f"  - RealLifeViolence: {len(rl_records)} videos")

    ucf_records = discover_ucf_crime(args.dataset_root) if "ucf" in selected else []
    print(f"  - UCF-Crime:        {len(ucf_records)} videos")

    st_records = discover_shanghaitech(args.dataset_root) if "shanghai" in selected else []
    print(f"  - ShanghaiTech:     {len(st_records)} videos")

    proc_records = discover_processed_clips(args.processed_root) if "processed" in selected else []
    print(f"  - Processed Clips:  {len(proc_records)} clips")

    raw_records = rl_records + ucf_records + st_records
    # Processed clips are derivatives of raw videos. Mixing both without a
    # provenance map leaks related content across splits, so prefer processed
    # clips when they are available.
    total_records = proc_records if proc_records else raw_records
    print(f"Total selected sequences: {len(total_records)}\n")

    print("Generating stratified splits (Train/Val/Test)...")

    # 1. Split for RealLifeViolence (fast benchmark)
    if rl_records:
        tr_v, va_v, te_v = split_records(rl_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        save_csv(args.output_dir / "train_violence.csv", tr_v)
        save_csv(args.output_dir / "val_violence.csv", va_v)
        save_csv(args.output_dir / "test_violence.csv", te_v)

    # 2. Split for UCF-Crime
    if ucf_records:
        tr_u, va_u, te_u = split_records(ucf_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        save_csv(args.output_dir / "train_ucf.csv", tr_u)
        save_csv(args.output_dir / "val_ucf.csv", va_u)
        save_csv(args.output_dir / "test_ucf.csv", te_u)

    # 3. Split for Processed Clips
    if proc_records:
        tr_p, va_p, te_p = split_records(proc_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        save_csv(args.output_dir / "train_processed.csv", tr_p)
        save_csv(args.output_dir / "val_processed.csv", va_p)
        save_csv(args.output_dir / "test_processed.csv", te_p)

    # 4. Primary default dataset (Processed clips if available, else RealLifeViolence)
    if proc_records:
        save_csv(args.output_dir / "train.csv", tr_p)
        save_csv(args.output_dir / "val.csv", va_p)
        save_csv(args.output_dir / "test.csv", te_p)
    elif rl_records:
        save_csv(args.output_dir / "train.csv", tr_v)
        save_csv(args.output_dir / "val.csv", va_v)
        save_csv(args.output_dir / "test.csv", te_v)

    # 5. Combined all dataset
    if total_records:
        tr_all, va_all, te_all = split_records(total_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        save_csv(args.output_dir / "train_all.csv", tr_all)
        save_csv(args.output_dir / "val_all.csv", va_all)
        save_csv(args.output_dir / "test_all.csv", te_all)

    elapsed = time.perf_counter() - start_time
    print("-" * 55)
    print(f"[OK] Manifest generation complete in {format_duration(elapsed)}!")
    print("=" * 55)


if __name__ == "__main__":
    main()


