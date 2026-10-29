"""
Multi-Seed Statistical Rigor Runner & 95% Confidence Interval Synthesizer (Phase 5).

Runs training and evaluation of Champion ARGUS Threat-5 across 3 independent seeds:
  Seeds: {42, 1337, 2026}
Computes sample mean, sample standard deviation, and 95% Confidence Intervals for:
  - In-Domain Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
  - Zero-Shot Cross-Dataset Metrics: UCF-Crime Frame-Level ROC-AUC, PR-AUC, EER, FAR
Saves consolidated results to outputs/seeds/multi_seed_summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_SEEDS = [42, 1337, 2026]


def compute_confidence_interval_95(values: list[float], use_students_t: bool = True) -> dict:
    """Compute mean, std, and 95% CI margin for n samples."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_margin": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0, "ci_margin": 0.0, "ci_low": round(mean, 4), "ci_high": round(mean, 4)}

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    # Critical value: Student's t with df=n-1 (t=4.303 for n=3) vs standard 1.96
    critical = 4.303 if (use_students_t and n == 3) else 1.96
    margin = critical * (std / math.sqrt(n))

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_margin": round(margin, 4),
        "ci_low": round(max(0.0, mean - margin), 4),
        "ci_high": round(min(1.0, mean + margin), 4),
        "formatted": f"{mean:.4f} ± {margin:.4f}",
    }


def _extract_ucf_metric(ucf_dict: dict, key: str) -> float:
    """Extract frame-level metric from either nested 'metrics' dict or top-level dict."""
    if isinstance(ucf_dict.get("metrics"), dict) and key in ucf_dict["metrics"]:
        return float(ucf_dict["metrics"][key])
    return float(ucf_dict[key])


def run_multi_seed(
    seeds: list[int] = DEFAULT_SEEDS,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_base: Path = PROJECT_ROOT / "outputs" / "seeds",
    skip_seed_42_if_exists: bool = True,
    device: str | None = None,
    eco_mode: bool = False,
) -> dict:
    import torch
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = output_base / "multi_seed_summary.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TEST_DEPS};{PROJECT_ROOT}"

    print("=" * 65)
    print("      ARGUS MULTI-SEED STATISTICAL RIGOR SUITE (PHASE 5)        ")
    print("=" * 65)
    print(f"Seeds:               {seeds}")
    print(f"Epochs per seed:     {epochs}")
    print(f"Batch size:          {batch_size}")
    print(f"Device:              {resolved_device.upper()}")
    print(f"Output directory:    {output_base}")
    print("=" * 65)

    seed_results = {}

    for seed in seeds:
        seed_dir = output_base / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = seed_dir / "best.pt"

        # Check existing Seed 42 checkpoint from main training run
        existing_champion = PROJECT_ROOT / "outputs" / "checkpoints_rlvd_full" / "best.pt"
        if seed == 42 and skip_seed_42_if_exists and existing_champion.exists() and not ckpt_path.exists():
            print(f"[INFO] Using verified Seed 42 checkpoint: {existing_champion}")
            import shutil
            shutil.copy2(existing_champion, ckpt_path)

        # 1. Train if checkpoint does not exist
        if not ckpt_path.exists():
            print(f"\n--- Training Seed {seed} from Scratch ({epochs} Epochs) ---")
            train_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "train.py"),
                "--config", str(PROJECT_ROOT / "configs" / "train_rlvd_full.yaml"),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--seed", str(seed),
                "--output-dir", str(seed_dir),
            ] + (["--eco-mode"] if eco_mode else []) + (["--device", resolved_device] if resolved_device else [])
            subprocess.run(train_cmd, env=env, check=True)

        # 2. In-Domain Evaluation on clean test set
        print(f"\n[EVAL] Evaluating Seed {seed} on clean held-out test split...")
        eval_in_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate.py"),
            "--checkpoint", str(ckpt_path),
            "--manifest", str(PROJECT_ROOT / "data" / "test_clean.csv"),
            "--output-json", str(seed_dir / "eval_test_clean.json"),
            "--allow-unsafe-checkpoint",
        ] + (["--device", resolved_device] if resolved_device else [])
        subprocess.run(eval_in_cmd, env=env, check=True)

        with open(seed_dir / "eval_test_clean.json") as f:
            in_metrics = json.load(f)

        # 3. Cross-Dataset Frame-Level Evaluation on UCF-Crime
        print(f"\n[EVAL] Evaluating Seed {seed} on UCF-Crime frame-level (111k frames)...")
        eval_ucf_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_frame_auc.py"),
            "--checkpoint", str(ckpt_path),
            "--output-json", str(seed_dir / "eval_ucf_frame_level.json"),
            "--streaming",
        ] + (["--device", resolved_device] if resolved_device else [])
        subprocess.run(eval_ucf_cmd, env=env, check=True)

        with open(seed_dir / "eval_ucf_frame_level.json") as f:
            ucf_metrics = json.load(f)

        seed_results[f"seed_{seed}"] = {
            "in_domain": in_metrics,
            "ucf_crime_frame_level": ucf_metrics,
        }

    # 4. Synthesize Confidence Intervals
    s_keys = list(seed_results.keys())
    roc_aucs_in = [seed_results[k]["in_domain"]["auc_roc"] for k in s_keys]
    f1s_in = [seed_results[k]["in_domain"]["f1"] for k in s_keys]
    accs_in = [seed_results[k]["in_domain"]["accuracy"] for k in s_keys]
    aps_in = [seed_results[k]["in_domain"]["average_precision"] for k in s_keys]

    ucf_aucs = [_extract_ucf_metric(seed_results[k]["ucf_crime_frame_level"], "frame_roc_auc") for k in s_keys]
    ucf_prs = [_extract_ucf_metric(seed_results[k]["ucf_crime_frame_level"], "frame_pr_auc") for k in s_keys]

    summary = {
        "seeds_evaluated": seeds,
        "n_seeds": len(seeds),
        "in_domain_95ci": {
            "roc_auc": compute_confidence_interval_95(roc_aucs_in),
            "f1_score": compute_confidence_interval_95(f1s_in),
            "accuracy": compute_confidence_interval_95(accs_in),
            "average_precision": compute_confidence_interval_95(aps_in),
        },
        "ucf_crime_zero_shot_95ci": {
            "frame_roc_auc": compute_confidence_interval_95(ucf_aucs),
            "frame_pr_auc": compute_confidence_interval_95(ucf_prs),
        },
        "per_seed_results": seed_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("         MULTI-SEED STATISTICAL RIGOR SUMMARY (TABLE III)        ")
    print("=" * 65)
    print(f"In-Domain ROC-AUC:   {summary['in_domain_95ci']['roc_auc']['formatted']}")
    print(f"In-Domain F1-Score:  {summary['in_domain_95ci']['f1_score']['formatted']}")
    print(f"In-Domain Accuracy:  {summary['in_domain_95ci']['accuracy']['formatted']}")
    print(f"UCF-Crime Zero-Shot: {summary['ucf_crime_zero_shot_95ci']['frame_roc_auc']['formatted']}")
    print("=" * 65)
    print(f"[OK] Summary JSON saved to: {summary_path}")

    return summary


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run multi-seed statistical rigor suite.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-base", type=Path, default=PROJECT_ROOT / "outputs" / "seeds")
    parser.add_argument("--device", type=str, default=None, help="Compute device override (cpu or cuda).")
    parser.add_argument("--retrain-seed-42", action="store_true", default=False)
    parser.add_argument("--eco-mode", action="store_true", default=False, help="Run in multitasking eco-mode")
    return parser.parse_args(args)


def main():
    args = parse_args()

    run_multi_seed(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_base=args.output_base,
        skip_seed_42_if_exists=not args.retrain_seed_42,
        device=args.device,
        eco_mode=args.eco_mode,
    )


if __name__ == "__main__":
    main()
