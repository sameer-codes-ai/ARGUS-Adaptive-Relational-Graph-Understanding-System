"""
Orchestrator for Step 3.2: UCF-Crime 290-video cross-dataset evaluation across all 6 ablation variants.
Safely skips already completed evaluations and logs progress.
"""

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    "ablation_1_no_graph",
    "ablation_2_no_edge_attrs",
    "ablation_3_no_roi_features",
    "ablation_4_no_pruning",
    "ablation_5_uniform_sampling",
    "ablation_6_no_causal_memory",
]

def main():
    print("=" * 70)
    print("   STAGE 3.2: CROSS-DATASET UCF-CRIME EVALUATION (ALL ABLATIONS)    ")
    print("=" * 70)

    for idx, variant in enumerate(VARIANTS, 1):
        ckpt_path = PROJECT_ROOT / "outputs" / "ablations" / variant / "best.pt"
        out_json = PROJECT_ROOT / "outputs" / "ablations" / variant / "eval_ucf_frame_level.json"

        print(f"\n[{idx}/{len(VARIANTS)}] Variant: {variant}")
        if out_json.is_file():
            print(f"  [SKIP] Already evaluated on UCF-Crime: {out_json}")
            continue

        if not ckpt_path.is_file():
            print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
            continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_frame_auc.py"),
            "--checkpoint", str(ckpt_path),
            "--output", str(out_json),
            "--streaming",
            "--device", "cuda",
        ]

        print(f"  [RUN] Evaluating {variant} on 290 UCF-Crime test videos (111,308 frames)...")
        start = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - start

        if res.returncode == 0 and out_json.is_file():
            print(f"  [DONE] {variant} evaluated successfully in {elapsed/60:.1f} min")
        else:
            print(f"  [FAILED] Evaluation failed for {variant} with exit code {res.returncode}")

    print("\n" + "=" * 70)
    print("   STAGE 3.2 EVALUATION RUN COMPLETE FOR ALL ABLATION VARIANTS    ")
    print("=" * 70)

if __name__ == "__main__":
    main()
