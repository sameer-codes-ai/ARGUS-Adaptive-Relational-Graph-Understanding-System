"""
Master Pipeline Runner for ARGUS Final Execution.
Sequentially executes:
  1. Stage 3.2: Complete remaining Ablation UCF-Crime evaluation (Ablation 6).
  2. Stage 4: Multi-Seed Statistical Rigor Suite (Seeds 42, 1337, 2026 with 95% CIs).
  3. Stage 5: Figure 4 Pareto Frontier Synthesis.
  4. Stage 5: Authoritative Results & Scientific Integrity Audit.
"""

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_step(step_name: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 70)
    print(f"  >>> RUNNING: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    if res.returncode == 0:
        print(f"\n[OK] {step_name} completed successfully in {elapsed/60:.1f} minutes!")
        return True
    else:
        print(f"\n[ERROR] {step_name} failed with exit code {res.returncode}")
        return False

def main():
    print("=" * 70)
    print("      ARGUS COMPLETE PIPELINE EXECUTION (STAGES 3.2 -> 5)       ")
    print("=" * 70)
    total_start = time.time()

    # 1. Complete Stage 3.2 (Ablation UCF-Crime evaluations)
    step1_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_ablation_ucf_evals.py"),
    ]
    if not run_step("Stage 3.2: Ablation UCF-Crime Frame Evaluations", step1_cmd):
        sys.exit(1)

    # 2. Complete Stage 4 (Multi-Seed Statistical Rigor Suite)
    step2_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_multi_seed.py"),
        "--seeds", "42", "1337", "2026",
        "--epochs", "15",
        "--batch-size", "8",
        "--device", "cuda",
        "--output-base", str(PROJECT_ROOT / "outputs" / "seeds"),
    ]
    if not run_step("Stage 4: Multi-Seed Statistical Rigor (Table III)", step2_cmd):
        sys.exit(1)

    # 3. Complete Stage 5 (Pareto Frontier Plots)
    step3_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_pareto_plots.py"),
        "--compile-from", str(PROJECT_ROOT / "outputs"),
        "--output", str(PROJECT_ROOT / "figures" / "pareto_analysis.png"),
    ]
    if not run_step("Stage 5: Pareto Analysis Plot & Metrics", step3_cmd):
        sys.exit(1)

    # 4. Complete Stage 5 (Authoritative Audit)
    step4_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "audit_results.py"),
    ]
    if not run_step("Stage 5: Authoritative Scientific Integrity Audit", step4_cmd):
        sys.exit(1)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("      ALL EXPERIMENTAL SUITE STAGES SUCCESSFULLY COMPLETED!     ")
    print(f"Total Execution Time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} mins)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
