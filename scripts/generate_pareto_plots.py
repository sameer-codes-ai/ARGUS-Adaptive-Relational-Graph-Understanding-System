"""Generate a Pareto plot exclusively from measured, supplied metrics.

The input JSON maps model names to records containing ``auc``, ``fps``,
``gflops``, ``params_m`` and ``memory_mb``. No benchmark defaults are supplied:
an incomplete experiment cannot become a publication-looking graph by accident.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_FIELDS = {"auc", "fps", "gflops", "params_m", "memory_mb"}
COLORS = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["*", "o", "s", "^", "v", "p", "h", "D", "X", "<"]


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Metrics JSON must be a non-empty object keyed by model name")
    parsed = {}
    for name, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"Metrics for {name!r} must be an object")
        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            raise ValueError(f"Metrics for {name!r} are missing: {sorted(missing)}")
        parsed[str(name)] = {key: float(item[key]) for key in REQUIRED_FIELDS}
    return parsed


def generate_pareto_plots(output_png: Path, metrics_json: Path) -> None:
    data = load_metrics(metrics_json)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    panels = [
        ("fps", "Throughput (Frames Per Second)", "(a) Accuracy vs. Inference Throughput"),
        ("gflops", "Complexity (GFLOPs / 16-Frame Clip)", "(b) Accuracy vs. Computational Complexity"),
        ("params_m", "Total Parameters (Millions)", "(c) Accuracy vs. Parameter Footprint"),
        ("memory_mb", "Peak Working Memory (MB)", "(d) Accuracy vs. Peak Working Memory"),
    ]
    for ax, (key, xlabel, title) in zip(axes.flat, panels):
        for index, (name, metrics) in enumerate(data.items()):
            ax.scatter(metrics[key], metrics["auc"], color=COLORS[index % len(COLORS)],
                       marker=MARKERS[index % len(MARKERS)], s=150 if index == 0 else 80,
                       label=name, edgecolors="black", linewidth=0.8, zorder=4)
        if key == "fps":
            ax.axvline(25.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
                       label="25 FPS reference")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("ROC-AUC Score", fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[OK] Measured-data Pareto plot generated: {output_png}")


def _extract_auc(data: dict, source: Path) -> float:
    auc = data.get("auc_roc", data.get("roc_auc", data.get("auc")))
    if auc is None:
        raise ValueError(f"Missing 'auc_roc'/'roc_auc' in {source}")
    return float(auc)


def _extract_model_profile(model_dir: Path, eval_data: dict, eval_path: Path) -> dict[str, float]:
    """Extract measured fps, gflops, params_m, and memory_mb for a baseline/ablation without defaults."""
    comp_data: dict = {}
    comp_path = model_dir / "complexity_profile.json"
    if comp_path.is_file():
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)

    mem_data: dict = {}
    mem_path = model_dir / "memory_profile.json"
    if mem_path.is_file():
        with open(mem_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)

    full_m = comp_data.get("full_model", {}) if isinstance(comp_data, dict) else {}
    fps = eval_data.get("fps", eval_data.get("throughput_fps", full_m.get("throughput_fps")))
    gflops = eval_data.get("gflops", eval_data.get("total_gflops_per_clip", full_m.get("total_gflops_per_clip")))
    params_m = eval_data.get("params_m", full_m.get("params_m"))
    if params_m is None and "total_parameters" in full_m:
        params_m = float(full_m["total_parameters"]) / 1e6
    memory_mb = eval_data.get("memory_mb", eval_data.get("peak_rss_mb", mem_data.get("peak_rss_mb")))

    missing = [
        name
        for name, val in (("fps", fps), ("gflops", gflops), ("params_m", params_m), ("memory_mb", memory_mb))
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Missing measured profiling metrics {missing} for {eval_path.parent.name} "
            f"(expected in {eval_path.name} or {model_dir}/complexity_profile.json & memory_profile.json)"
        )

    return {
        "auc": _extract_auc(eval_data, eval_path),
        "fps": float(fps),
        "gflops": float(gflops),
        "params_m": round(float(params_m), 2),
        "memory_mb": float(memory_mb),
    }


def compile_pareto_metrics(compile_dir: Path, output_json: Path | None = None) -> Path:
    """Compile verified measured metrics across outputs/ into a unified Pareto JSON."""
    compile_dir = Path(compile_dir)
    target_json = output_json or (compile_dir / "pareto_metrics.json")

    # If target already exists and passes schema validation, return it directly
    if target_json.is_file():
        load_metrics(target_json)
        return target_json

    # 1. Ingest Complexity Profiling
    comp_path = compile_dir / "profiling" / "complexity_profile.json"
    if not comp_path.is_file():
        raise FileNotFoundError(f"Required complexity profile not found: {comp_path}")
    with open(comp_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    # 2. Ingest Memory Profiling
    mem_path = compile_dir / "profiling" / "memory_profile.json"
    if not mem_path.is_file():
        raise FileNotFoundError(f"Required memory profile not found: {mem_path}")
    with open(mem_path, "r", encoding="utf-8") as f:
        mem_data = json.load(f)

    # 3. Ingest Champion Evaluation
    champ_eval = compile_dir / "checkpoints_rlvd_full" / "eval_test_clean.json"
    if not champ_eval.is_file():
        raise FileNotFoundError(f"Required champion evaluation file not found: {champ_eval}")
    with open(champ_eval, "r", encoding="utf-8") as f:
        c_data = json.load(f)
    champ_auc = _extract_auc(c_data, champ_eval)

    full_m = comp_data.get("full_model")
    if not isinstance(full_m, dict):
        raise ValueError(f"Missing 'full_model' section in {comp_path}")
    if "throughput_fps" not in full_m or "total_gflops_per_clip" not in full_m:
        raise ValueError(f"Missing 'throughput_fps' or 'total_gflops_per_clip' in {comp_path}")
    if "params_m" in full_m:
        champ_params = float(full_m["params_m"])
    elif "total_parameters" in full_m:
        champ_params = round(float(full_m["total_parameters"]) / 1e6, 2)
    else:
        raise ValueError(f"Missing 'total_parameters'/'params_m' in {comp_path}")

    if "peak_rss_mb" not in mem_data or mem_data["peak_rss_mb"] is None:
        raise ValueError(f"Missing 'peak_rss_mb' in {mem_path}")

    metrics: dict[str, dict[str, float]] = {
        "ARGUS Threat-5 (Champion)": {
            "auc": champ_auc,
            "fps": float(full_m["throughput_fps"]),
            "gflops": float(full_m["total_gflops_per_clip"]),
            "params_m": champ_params,
            "memory_mb": float(mem_data["peak_rss_mb"]),
        }
    }

    # 4. Ingest Baselines (required)
    baseline_specs = {
        "convnext_mil": "ConvNeXt-Tiny + MIL",
        "causal_tsm": "Causal TSM",
        "rtfm_lite": "RTFM-Lite",
    }

    for b_key, disp_name in baseline_specs.items():
        b_dir = compile_dir / "baselines" / b_key
        b_eval = b_dir / "eval_test_clean.json"
        if not b_eval.is_file():
            raise FileNotFoundError(f"Required baseline evaluation file not found: {b_eval}")
        with open(b_eval, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        metrics[disp_name] = _extract_model_profile(b_dir, b_data, b_eval)

    # 5. Ingest Ablations (if present)
    ablation_specs = {
        "ablation_1_no_graph": "Ablation 1 (No Graph)",
        "ablation_2_no_edge_attrs": "Ablation 2 (No Edge Attr)",
        "ablation_3_no_roi_features": "Ablation 3 (No RoI Feat)",
        "ablation_4_no_pruning": "Ablation 4 (No Pruning)",
        "ablation_5_uniform_sampling": "Ablation 5 (Uniform Stride)",
        "ablation_6_no_causal_memory": "Ablation 6 (No Memory)",
    }

    for a_key, disp_name in ablation_specs.items():
        a_dir = compile_dir / "ablations" / a_key
        a_eval = a_dir / "eval_test_clean.json"
        if a_eval.is_file():
            with open(a_eval, "r", encoding="utf-8") as f:
                a_data = json.load(f)
            metrics[disp_name] = _extract_model_profile(a_dir, a_data, a_eval)

    target_json.parent.mkdir(parents=True, exist_ok=True)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    load_metrics(target_json)
    print(f"[OK] Compiled {len(metrics)} model records to {target_json}")
    return target_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto plots from measured metrics JSON.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Validated measured metric records.")
    parser.add_argument("--compile-from", type=Path, default=None, help="Directory to compile metrics from.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "pareto_analysis.png")
    args = parser.parse_args()

    metrics_path = args.metrics_json
    if metrics_path is None:
        compile_root = args.compile_from or (PROJECT_ROOT / "outputs")
        metrics_path = compile_pareto_metrics(compile_root)

    generate_pareto_plots(args.output, metrics_path)


if __name__ == "__main__":
    main()
