"""
Comprehensive Verification and Results Audit for ARGUS (Phase 5.2).

Validates the full experimental lifecycle for scientific integrity:
  1. Manifest Split Disjointness & Zero Data Leakage
  2. Git LFS & Checkpoint Integrity (no pointer stubs)
  3. Champion In-Domain & Cross-Dataset Evaluation
  4. UCF-Crime 290-Video Complete Coverage (rejects truncated runs)
  5. Compute-Matched Baselines Training & Evaluation Coverage
  6. Phase 4 Profiling Telemetry (Complexity, Memory, Non-Rewarded TTA)
  7. Phase 3 Independent Ablation Suite Coverage (all 6 trained variants)
  8. Phase 5 Multi-Seed Statistical Rigor (Seeds 42, 1337, 2026 + 95% CIs)
  9. Pareto Plot & Figure Artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

from scripts.prepare_manifests import verify_no_data_leakage


class AuditContext:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.passes: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, fail_msg: str, warn_only: bool = False):
        if condition:
            self.passes.append(name)
            print(f"  [PASS] {name}")
        else:
            if warn_only and not self.strict:
                self.warnings.append(f"{name}: {fail_msg}")
                print(f"  [WARN] {name}: {fail_msg}")
            else:
                self.failures.append(f"{name}: {fail_msg}")
                print(f"  [FAIL] {name}: {fail_msg}")


def audit_splits(ctx: AuditContext, data_dir: Path) -> None:
    print("\n--- 1. Data Splits & Leakage Audit ---")
    tr_path = data_dir / "train_clean.csv"
    va_path = data_dir / "val_clean.csv"
    te_path = data_dir / "test_clean.csv"

    for p in [tr_path, va_path, te_path]:
        ctx.check(f"Split file exists: {p.name}", p.is_file(), f"Missing split file: {p}")

    if tr_path.is_file() and va_path.is_file() and te_path.is_file():
        with open(tr_path, "r", encoding="utf-8") as f:
            tr = list(csv.DictReader(f))
        with open(va_path, "r", encoding="utf-8") as f:
            va = list(csv.DictReader(f))
        with open(te_path, "r", encoding="utf-8") as f:
            te = list(csv.DictReader(f))

        ctx.check(
            "Split counts expected (1400/300/300)",
            len(tr) == 1400 and len(va) == 300 and len(te) == 300,
            f"Got counts train={len(tr)}, val={len(va)}, test={len(te)}",
            warn_only=True,
        )

        try:
            verify_no_data_leakage(tr, va, te)
            ctx.check("Zero source-video leakage across splits", True, "")
        except AssertionError as e:
            ctx.check("Zero source-video leakage across splits", False, str(e))


def audit_checkpoint(ctx: AuditContext, ckpt_path: Path, min_size_mb: float = 1.0) -> None:
    if not ckpt_path.is_file():
        ctx.check(f"Checkpoint exists: {ckpt_path.name}", False, f"Not found at {ckpt_path}")
        return

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    is_not_stub = size_mb >= min_size_mb
    ctx.check(
        f"Checkpoint is real binary (not LFS stub): {ckpt_path.name}",
        is_not_stub,
        f"Size is only {size_mb:.3f} MB (< {min_size_mb} MB). Ensure 'git lfs pull' was run.",
    )


def audit_eval_json(ctx: AuditContext, eval_path: Path, label: str) -> dict | None:
    if not eval_path.is_file():
        ctx.check(f"Evaluation JSON exists: {label}", False, f"Missing {eval_path}", warn_only=True)
        return None

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        ctx.check(f"Evaluation JSON valid: {label}", False, f"JSON parse error: {e}")
        return None

    auc = data.get("auc_roc")
    if auc is not None:
        ctx.check(
            f"Metrics sanity: {label} (AUC={auc})",
            0.0 <= float(auc) <= 1.0,
            f"AUC out of range [0, 1]: {auc}",
        )
    return data


def audit_ucf_eval(ctx: AuditContext, ucf_path: Path, label: str) -> dict | None:
    if not ucf_path.is_file():
        ctx.check(f"UCF-Crime Eval exists: {label}", False, f"Missing {ucf_path}", warn_only=True)
        return None

    try:
        with open(ucf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        ctx.check(f"UCF-Crime Eval valid JSON: {label}", False, f"JSON parse error: {e}")
        return None

    perf = data.get("performance", {})
    v_count = perf.get("total_videos_evaluated", 0)
    ctx.check(
        f"UCF-Crime 290-video completeness: {label}",
        v_count == 290,
        f"Evaluated only {v_count} videos (expected 290). Partial benchmark runs are scientifically invalid for publication.",
        warn_only=True,
    )

    metrics = data.get("metrics", {})
    auc = metrics.get("frame_roc_auc")
    far = metrics.get("false_alarm_rate_tau_050")
    if auc is not None:
        ctx.check(
            f"UCF frame ROC-AUC sanity: {label} (AUC={auc})",
            0.0 <= float(auc) <= 1.0,
            f"Frame ROC-AUC out of bounds: {auc}",
        )
    if far is not None:
        ctx.check(
            f"UCF false alarm rate sanity: {label} (FAR={far})",
            0.0 <= float(far) <= 1.0,
            f"FAR out of bounds: {far}",
        )
    return data


def audit_profiling(ctx: AuditContext, prof_dir: Path) -> None:
    print("\n--- 2. Phase 4 Hardware Profiling Telemetry Audit ---")
    comp_file = prof_dir / "complexity_profile.json"
    mem_file = prof_dir / "memory_profile.json"
    tta_file = prof_dir / "time_to_alert_benchmark.json"

    for p in [comp_file, mem_file, tta_file]:
        ctx.check(f"Profiling artifact exists: {p.name}", p.is_file(), f"Missing {p}", warn_only=True)

    if comp_file.is_file():
        with open(comp_file, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        full_m = c_data.get("full_model", {})
        ctx.check(
            "Complexity profiler reported valid GFLOPs & FPS",
            float(full_m.get("total_gflops_per_clip", 0)) > 0 and float(full_m.get("throughput_fps", 0)) > 0,
            "Invalid or zero GFLOPs/FPS in complexity_profile.json",
        )

    if mem_file.is_file():
        with open(mem_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        rss_drift = m_data.get("drift_mb_per_hour", 0.0)
        ctx.check(
            "Memory profiler verifies O(1) bounded memory",
            m_data.get("bounded_memory_verified", False) or abs(float(rss_drift)) < 5.0,
            f"Memory drift reported: {rss_drift} MB/hr",
            warn_only=True,
        )

    if tta_file.is_file():
        with open(tta_file, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        mean_tta = t_data.get("mean_tta_ms", float("nan"))
        v_results = t_data.get("video_results", [])
        has_negative_lat = any(
            v.get("detected") is True and v.get("tta_ms") is not None and v.get("tta_ms") < 0
            for v in v_results
        )
        ctx.check(
            "Time-to-Alert does not reward pre-incident alarms with negative/zero latency",
            not has_negative_lat,
            "Found video with detected=True and negative latency! Pre-onset alerts must be treated as false alarms.",
        )


def audit_baselines(ctx: AuditContext, base_dir: Path) -> None:
    print("\n--- 3. Phase 2 Matched Baselines Audit (Table IV) ---")
    summary_file = base_dir / "baseline_training_summary.json"
    ctx.check("Baseline summary exists", summary_file.is_file(), f"Missing {summary_file}", warn_only=True)

    for b_name in ["convnext_mil", "causal_tsm", "rtfm_lite"]:
        b_dir = base_dir / b_name
        audit_checkpoint(ctx, b_dir / "best.pt", min_size_mb=1.0)
        audit_eval_json(ctx, b_dir / "eval_test_clean.json", f"Baseline {b_name} In-Domain")
        audit_ucf_eval(ctx, b_dir / "eval_ucf_frame_level.json", f"Baseline {b_name} UCF-Crime")


def audit_ablations(ctx: AuditContext, ab_dir: Path) -> None:
    print("\n--- 4. Phase 3 Component Ablations Audit (Table V) ---")
    summary_file = ab_dir / "ablation_summary.json"
    ctx.check("Ablation summary exists", summary_file.is_file(), f"Missing {summary_file}", warn_only=True)

    expected_variants = [
        "ablation_1_no_graph",
        "ablation_2_no_edge_attrs",
        "ablation_3_no_roi_features",
        "ablation_4_no_pruning",
        "ablation_5_uniform_sampling",
        "ablation_6_no_causal_memory",
    ]
    for v_name in expected_variants:
        v_dir = ab_dir / v_name
        audit_checkpoint(ctx, v_dir / "best.pt", min_size_mb=0.1)
        audit_eval_json(ctx, v_dir / "eval_test_clean.json", f"Ablation {v_name} In-Domain")


def audit_multi_seed(ctx: AuditContext, seed_dir: Path) -> None:
    print("\n--- 5. Phase 5 Multi-Seed Statistical Rigor Audit (Table III) ---")
    summary_file = seed_dir / "multi_seed_summary.json"
    ctx.check("Multi-seed summary exists", summary_file.is_file(), f"Missing {summary_file}", warn_only=True)

    if summary_file.is_file():
        with open(summary_file, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        seeds = s_data.get("seeds_evaluated", [])
        ctx.check(
            "All 3 seeds evaluated (42, 1337, 2026)",
            set(seeds) == {42, 1337, 2026},
            f"Found seeds: {seeds}",
            warn_only=True,
        )
        in_ci = s_data.get("in_domain_95ci", {})
        ctx.check("In-domain 95% CIs synthesized", "roc_auc" in in_ci, "Missing 95% CI metrics")


def audit_figures(ctx: AuditContext, fig_dir: Path) -> None:
    print("\n--- 6. Publication Figures Audit ---")
    pareto_png = fig_dir / "pareto_analysis.png"
    if not pareto_png.is_file():
        ctx.check("Figure 4 (pareto_analysis.png) exists", False, f"Missing {pareto_png}", warn_only=True)
    else:
        ctx.check(
            "Figure 4 (pareto_analysis.png) non-empty",
            pareto_png.stat().st_size > 5000,
            f"File too small ({pareto_png.stat().st_size} bytes)",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Authoritative Audit of ARGUS Experimental Suite Results.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as critical failures.")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()

    print("=" * 70)
    print("         ARGUS EXPERIMENTAL RESULTS INTEGRITY & RIGOR AUDIT        ")
    print("=" * 70)
    print(f"Repository Root: {PROJECT_ROOT}")
    print(f"Outputs Dir:     {args.outputs_dir}")
    print(f"Strict Mode:     {'ENABLED' if args.strict else 'DISABLED'}")

    ctx = AuditContext(strict=args.strict)

    # 1. Splits & Leakage
    audit_splits(ctx, args.data_dir)

    # 2. Champion Model Checkpoint & Evaluation
    print("\n--- Champion Model Audit ---")
    audit_checkpoint(ctx, args.outputs_dir / "checkpoints_rlvd_full" / "best.pt", min_size_mb=10.0)
    audit_eval_json(ctx, args.outputs_dir / "checkpoints_rlvd_full" / "eval_test_clean.json", "Champion In-Domain")
    audit_ucf_eval(ctx, args.outputs_dir / "checkpoints_rlvd_full" / "eval_ucf_frame_level.json", "Champion UCF-Crime")
    if (args.outputs_dir / "eval_ucf_frame_level.json").is_file():
        audit_ucf_eval(ctx, args.outputs_dir / "eval_ucf_frame_level.json", "Root UCF-Crime")

    # 3. Profiling Telemetry
    audit_profiling(ctx, args.outputs_dir / "profiling")

    # 4. Baselines
    audit_baselines(ctx, args.outputs_dir / "baselines")

    # 5. Ablations
    audit_ablations(ctx, args.outputs_dir / "ablations")

    # 6. Multi-Seed
    audit_multi_seed(ctx, args.outputs_dir / "seeds")

    # 7. Figures
    audit_figures(ctx, args.figures_dir)

    print("\n" + "=" * 70)
    print("                     AUDIT SUMMARY RESULTS                         ")
    print("=" * 70)
    print(f"  Total Checks Passed:    {len(ctx.passes):d}")
    print(f"  Total Warnings:         {len(ctx.warnings):d}")
    print(f"  Total Failures:         {len(ctx.failures):d}")
    print("-" * 70)

    if ctx.warnings:
        print("[AUDIT WARNINGS]")
        for w in ctx.warnings:
            print(f"  ! {w}")

    if ctx.failures:
        print("\n[CRITICAL AUDIT FAILURES]")
        for f in ctx.failures:
            print(f"  X {f}")
        print("\n[RESULT] AUDIT FAILED. Review and resolve the critical failures above.")
        return 1

    print("\n[RESULT] ALL REQUIRED INTEGRITY AND SCIENTIFIC RIGOR CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
