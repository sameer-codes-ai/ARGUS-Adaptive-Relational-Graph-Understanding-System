"""
Controlled Component Ablation Suite Runner for ARGUS (Phase 3).

Executes the 6 ablation variants defined in Plan.md:
  1. ablation_1_no_graph: ConvNeXt tokens directly to Temporal Transformer (no graph, no detector).
  2. ablation_2_no_edge_attrs: Standard GAT without 10D relational/threat edge attributes.
  3. ablation_3_no_roi_features: Scene graph nodes have 12D kinematic/semantic one-hot only (no 64D RoI visual crops).
  4. ablation_4_no_pruning: All visual tokens retained in transformer (keep_ratio: 1.0).
  5. ablation_5_uniform_sampling: Fixed stride uniform frame selection (motion entropy disabled).
  6. ablation_6_no_causal_memory: Single-clip context, historical FIFO memory disabled (memory_size: 0).

Supports:
  - Full 15-epoch retraining from scratch across any/all variants
  - Direct zero-shot evaluation (--eval-only) of the champion model under ablated inference conditions
  - Consolidated JSON metrics output to outputs/ablations/ablation_summary.json
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
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

DEFAULT_VARIANTS = [
    "ablation_1_no_graph",
    "ablation_2_no_edge_attrs",
    "ablation_3_no_roi_features",
    "ablation_4_no_pruning",
    "ablation_5_uniform_sampling",
    "ablation_6_no_causal_memory",
]


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
    parser = argparse.ArgumentParser(description="Run ARGUS component ablation suite.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=DEFAULT_VARIANTS,
        help="List of ablation variants to run.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablations",
        help="Directory containing ablation YAML files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per ablation variant (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ablations",
        help="Base directory to save ablation checkpoints and logs.",
    )
    parser.add_argument(
        "--eco-mode",
        action="store_true",
        default=False,
        help="Run training in daily-multitasking eco-mode (caps CPU threads to 4, sets BelowNormal priority, caps VRAM).",
    )
    parser.add_argument(
        "--champion-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints_rlvd_full" / "best.pt",
        help="Path to trained ARGUS champion checkpoint.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_clean.csv",
        help="Path to clean test manifest CSV.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on test split without training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cpu or cuda).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Force retraining from scratch even if variant already completed.",
    )
    return parser.parse_args(args)


def run_training_for_variant(
    variant: str,
    config_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    eco_mode: bool = False,
    device: str | None = None,
    force_retrain: bool = False,
) -> dict:
    """Run training for a single ablation variant using train.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best.pt"
    history_file = output_dir / "training_history.json"

    if not force_retrain and best_ckpt.is_file() and history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                existing_history = json.load(f)
            if isinstance(existing_history, list) and len(existing_history) >= epochs:
                best_entry = max(existing_history, key=lambda x: x.get("val_metrics", {}).get("auc_roc", 0.0))
                total_dur = sum(float(h.get("epoch_duration_s", 0.0)) for h in existing_history)
                best_metrics = {
                    "best_epoch": best_entry.get("epoch"),
                    "val_auc_roc": best_entry.get("val_metrics", {}).get("auc_roc"),
                    "val_ap": best_entry.get("val_metrics", {}).get("average_precision"),
                    "val_f1": best_entry.get("val_metrics", {}).get("f1"),
                    "val_loss": best_entry.get("val_loss"),
                }
                print(f"  [SKIP] {variant} already completed {len(existing_history)} epochs (Best Val AUC: {best_metrics['val_auc_roc']}).")
                return {
                    "status": "SUCCESS",
                    "duration_seconds": round(total_dur, 2),
                    "duration_formatted": format_duration(total_dur),
                    "best_checkpoint": str(best_ckpt),
                    "best_metrics": best_metrics,
                    "note": f"Loaded from existing {len(existing_history)}-epoch checkpoint",
                }
        except Exception:
            pass

    python_exe = sys.executable

    cmd = [
        python_exe,
        str(PROJECT_ROOT / "train.py"),
        "--config", str(config_path),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--output-dir", str(output_dir),
    ] + (["--eco-mode"] if eco_mode else []) + (["--device", device] if device else [])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TEST_DEPS};{PROJECT_ROOT}"

    print("\n" + "=" * 70)
    print(f"  LAUNCHING TRAINING FOR: {variant}")
    print(f"  Config:     {config_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Recipe:     {epochs} Epochs | Batch {batch_size} | LR {lr} | AdamW")
    print("=" * 70 + "\n")

    start_time = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] Training failed for {variant} with exit code {result.returncode}")
        return {"status": "FAILED", "duration_seconds": elapsed}

    # Ingest training history
    history_file = output_dir / "training_history.json"
    best_metrics = {}
    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                best_entry = max(history, key=lambda x: x.get("val_metrics", {}).get("auc_roc", 0.0))
                best_metrics = {
                    "best_epoch": best_entry.get("epoch"),
                    "val_auc_roc": best_entry.get("val_metrics", {}).get("auc_roc"),
                    "val_ap": best_entry.get("val_metrics", {}).get("average_precision"),
                    "val_f1": best_entry.get("val_metrics", {}).get("f1"),
                    "val_loss": best_entry.get("val_loss"),
                }
        except Exception as e:
            print(f"[WARNING] Could not parse history for {variant}: {e}")

    return {
        "status": "SUCCESS",
        "duration_seconds": round(elapsed, 2),
        "duration_formatted": format_duration(elapsed),
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_metrics": best_metrics,
    }


def evaluate_variant(
    variant: str,
    checkpoint_path: Path,
    test_manifest: Path,
    config_path: Path,
    output_dir: Path,
    device: str | None = None,
) -> dict:
    """Run clip-level test evaluation using evaluate.py."""
    if not checkpoint_path.is_file():
        return {"eval_status": "SKIPPED (Checkpoint not found)"}

    eval_json = output_dir / f"eval_{variant}_test.json"
    python_exe = sys.executable

    cmd = [
        python_exe,
        str(PROJECT_ROOT / "evaluate.py"),
        "--checkpoint", str(checkpoint_path),
        "--manifest", str(test_manifest),
        "--output-json", str(eval_json),
        "--allow-unsafe-checkpoint",
    ]
    if config_path.is_file():
        cmd.extend(["--config", str(config_path)])
    if device:
        cmd.extend(["--device", device])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TEST_DEPS};{PROJECT_ROOT}"

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode == 0 and eval_json.is_file():
        try:
            with open(eval_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "test_roc_auc": data.get("auc_roc"),
                "test_ap": data.get("average_precision"),
                "test_f1": data.get("f1"),
                "test_accuracy": data.get("accuracy"),
                "fps": data.get("fps"),
                "latency_ms": data.get("latency_ms_per_clip"),
            }
        except Exception:
            pass
    return {"eval_status": "ERROR during evaluation"}


def main():
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("     ARGUS CONTROLLED COMPONENT ABLATION RUNNER (PHASE 3)      ")
    print("=" * 70)
    print(f"Variants Selected: {', '.join(args.variants)}")
    print(f"Configs Dir:       {args.configs_dir}")
    print(f"Output Base:       {args.output_base}")
    print(f"Mode:              {'EVAL-ONLY' if args.eval_only else f'TRAINING ({args.epochs} Epochs)'}")
    print("=" * 70)

    summary_results: dict[str, dict] = {}
    total_start = time.time()

    for idx, variant in enumerate(args.variants, 1):
        print(f"\n[{idx}/{len(args.variants)}] Processing: {variant}...")
        config_path = args.configs_dir / f"{variant}.yaml"
        output_dir = args.output_base / variant

        if not config_path.is_file():
            print(f"[ERROR] Config file not found: {config_path}")
            summary_results[variant] = {"status": "CONFIG_NOT_FOUND"}
            continue

        if args.eval_only:
            # In eval-only mode, use existing ablation checkpoint or export ablated champion checkpoint
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt = output_dir / "best.pt"
            if not ckpt.is_file() and args.champion_checkpoint.is_file():
                try:
                    import torch
                    import yaml
                    from argus.checkpoint import load_checkpoint
                    state = load_checkpoint(args.champion_checkpoint, map_location="cpu", allow_unsafe=True)
                    with open(config_path, "r", encoding="utf-8") as f:
                        yaml_cfg = yaml.safe_load(f) or {}
                    if isinstance(state, dict) and isinstance(yaml_cfg.get("model"), dict):
                        merged_cfg = dict(state.get("config", {}) or {})
                        merged_cfg.update(yaml_cfg["model"])
                        state["config"] = merged_cfg
                        torch.save(state, ckpt)
                except Exception:
                    ckpt = args.champion_checkpoint
            elif not ckpt.is_file():
                ckpt = args.champion_checkpoint
            eval_res = evaluate_variant(variant, ckpt, args.test_manifest, config_path, output_dir, device=args.device)
            summary_results[variant] = {"eval_metrics": eval_res, "checkpoint_evaluated": str(ckpt)}
        else:
            # Training mode
            train_res = run_training_for_variant(
                variant=variant,
                config_path=config_path,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                eco_mode=args.eco_mode,
                device=args.device,
                force_retrain=args.force_retrain,
            )
            # Evaluate after training
            if train_res.get("status") == "SUCCESS":
                eval_res = evaluate_variant(
                    variant,
                    output_dir / "best.pt",
                    args.test_manifest,
                    config_path,
                    output_dir,
                    device=args.device,
                )
                train_res["test_metrics"] = eval_res
            summary_results[variant] = train_res

    total_duration = time.time() - total_start

    # Consolidated JSON output
    summary_file = args.output_base / "ablation_summary.json"
    consolidated = {
        "phase": "Phase 3: Controlled Component Ablation Suite",
        "total_variants_executed": len(args.variants),
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_formatted": format_duration(total_duration),
        "results": summary_results,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print("\n" + "=" * 70)
    print("             ABLATION SUITE RUN COMPLETE               ")
    print(f"Total Duration:     {format_duration(total_duration)}")
    print(f"Consolidated Report:{summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
