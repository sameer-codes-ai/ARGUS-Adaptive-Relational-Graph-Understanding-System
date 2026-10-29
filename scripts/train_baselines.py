"""
Sequential Baseline Training Runner for ARGUS (Phase 2).
Executes the identical 15-epoch training recipe on the clean RLVD dataset split across:
  1. ConvNeXt-Tiny + MIL (convnext_mil)
  2. Causal Streaming TSM (causal_tsm)
  3. Lightweight RTFM (rtfm_lite)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODELS = ["convnext_mil", "causal_tsm", "rtfm_lite"]


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train compute-matched surveillance baselines.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="List of baseline models to train.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per baseline.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--train-manifest", type=str, default="data/train_clean.csv", help="Path to clean train CSV.")
    parser.add_argument("--val-manifest", type=str, default="data/val_clean.csv", help="Path to clean validation CSV.")
    parser.add_argument("--output-base", type=Path, default=Path("outputs/baselines"), help="Root directory for baseline outputs.")
    parser.add_argument("--device", type=str, default=None, help="Compute device override.")
    parser.add_argument("--eco-mode", action="store_true", default=False, help="Run in daily multitasking eco-mode.")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches for quick testing.")
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retraining even if checkpoint already completed.")
    return parser.parse_args(args)


def main():
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      ARGUS PHASE 2: COMPUTE-MATCHED BASELINE TRAINING SUITE    ")
    print("=" * 65)
    print(f"Target Models:       {', '.join(args.models)}")
    print(f"Epochs per model:    {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Learning rate:       {args.lr}")
    print(f"Train Manifest:      {args.train_manifest}")
    print(f"Val Manifest:        {args.val_manifest}")
    print(f"Output Root:         {args.output_base}")
    print("=" * 65)

    summary_records = {}
    suite_start = time.perf_counter()

    for idx, model_name in enumerate(args.models, start=1):
        model_out_dir = args.output_base / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = model_out_dir / "best.pt"
        history_file = model_out_dir / "training_history.json"

        if not args.force_retrain and best_ckpt.is_file() and history_file.is_file():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
                if isinstance(existing_history, list) and len(existing_history) >= args.epochs:
                    aucs = [h.get("val_metrics", {}).get("auc_roc") for h in existing_history if h.get("val_metrics")]
                    best_auc = max([a for a in aucs if a is not None], default=None)
                    total_dur = sum(float(h.get("epoch_duration_s", 0.0)) for h in existing_history)
                    summary_records[model_name] = {
                        "status": "SUCCESS",
                        "elapsed_seconds": round(total_dur, 2),
                        "formatted_time": format_duration(total_dur),
                        "best_checkpoint": str(best_ckpt),
                        "best_val_auc": best_auc,
                        "note": f"Loaded from existing {len(existing_history)}-epoch checkpoint",
                    }
                    print(f"\n[{idx}/{len(args.models)}] Skipping {model_name.upper()} (already completed {len(existing_history)} epochs, Best Val AUC: {best_auc})")
                    continue
            except Exception:
                pass

        print(f"\n[{idx}/{len(args.models)}] Training Baseline: {model_name.upper()}...")
        print(f"  Target directory: {model_out_dir}")

        cmd = [
            sys.executable,
            "train.py",
            "--model-type", model_name,
            "--manifest", str(args.train_manifest),
            "--val-manifest", str(args.val_manifest),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--output-dir", str(model_out_dir),
        ] + (["--device", args.device] if args.device else []) + (["--eco-mode"] if args.eco_mode else [])
        if args.max_batches is not None:
            cmd.extend(["--max-batches", str(args.max_batches)])

        env = os.environ.copy()
        testdeps = str(PROJECT_ROOT / ".testdeps")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{testdeps};{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = testdeps

        t0 = time.perf_counter()
        ret = subprocess.run(cmd, env=env)
        elapsed = time.perf_counter() - t0

        if ret.returncode != 0:
            print(f"[ERROR] Training failed for {model_name} with return code {ret.returncode}")
            summary_records[model_name] = {"status": "FAILED", "returncode": ret.returncode}
            continue

        best_ckpt = model_out_dir / "best.pt"
        history_file = model_out_dir / "training_history.json"
        best_auc = None

        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            aucs = [h.get("val_metrics", {}).get("auc_roc") for h in history if h.get("val_metrics")]
            best_auc = max([a for a in aucs if a is not None], default=None)

        summary_records[model_name] = {
            "status": "SUCCESS",
            "elapsed_seconds": round(elapsed, 2),
            "formatted_time": format_duration(elapsed),
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
            "best_val_auc": best_auc,
        }
        print(f"  [*] Completed {model_name.upper()} in {format_duration(elapsed)} (Best Val AUC: {best_auc})")

    suite_elapsed = time.perf_counter() - suite_start
    summary_path = args.output_base / "baseline_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite_duration_seconds": round(suite_elapsed, 2),
                "suite_duration_formatted": format_duration(suite_elapsed),
                "baselines": summary_records,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("               BASELINE TRAINING SUITE COMPLETED                ")
    print("=" * 65)
    print(f"Total Suite Duration: {format_duration(suite_elapsed)}")
    print(f"Summary JSON saved to: {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
