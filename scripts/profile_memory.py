"""
Empirical Constant-Memory Profiler for ARGUS (Phase 4.2).

Streams 10,000 continuous frames through ARGUS in streaming mode (stream=True)
with internal FIFO memory, sampling process RSS memory and PyTorch allocated
memory every 50 frames to empirically prove the O(1) memory invariance claim.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

import torch

try:
    import psutil
except ImportError:
    psutil = None

from argus.config import ARGUSConfig
from argus.model import ARGUS
from argus.checkpoint import load_checkpoint, model_state_dict


def get_process_memory_mb() -> float:
    """Return process RSS memory in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024.0 * 1024.0), 2)
    return 0.0


def profile_memory_stream(
    checkpoint_path: Path | None = None,
    total_frames: int = 10000,
    sample_interval: int = 50,
    device: str = "cpu",
    output_json: Path | None = None,
    output_plot: Path | None = None,
) -> dict:
    device = torch.device(device)
    print("=" * 65)
    print("        ARGUS EMPIRICAL O(1) STREAMING MEMORY PROFILER (PHASE 4.2)   ")
    print("=" * 65)
    print(f"Target Stream:       {total_frames:,} continuous frames")
    print(f"Sampling Interval:   Every {sample_interval} frames")
    print(f"Compute Device:      {device.type.upper()}")

    cfg = ARGUSConfig()
    if checkpoint_path and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path, map_location=device, allow_unsafe=True)
        if "config" in state:
            cfg = ARGUSConfig.from_dict(state["config"], strict=True)
        model = ARGUS(cfg).to(device)
        model.load_state_dict(model_state_dict(state), strict=False)
    else:
        model = ARGUS(cfg).to(device)

    model.eval()

    # Pre-generate 16 frames of noise to simulate real-time camera ingestion
    clip_len = 16
    warmup_clip = torch.randn(1, clip_len, 3, cfg.image_size, cfg.image_size, device=device)

    # Reset streaming FIFO memory
    if hasattr(model, "memory") and model.memory is not None:
        model.memory.reset(stream_id="stream_0")

    # Initial Warmup
    print("[INFO] Warming up model streaming pipeline...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(warmup_clip, stream=True, stream_id="stream_0")

    initial_rss_mb = get_process_memory_mb()
    print(f"[INFO] Baseline Process RSS Memory: {initial_rss_mb:.2f} MB")

    sampled_points = []
    frames_processed = 0
    start_time = time.perf_counter()

    # Stream frames through the model in batches of 16 (sliding window stride=16)
    num_steps = total_frames // clip_len

    for step in range(1, num_steps + 1):
        # Generate incoming video chunk
        incoming_clip = torch.randn(1, clip_len, 3, cfg.image_size, cfg.image_size, device=device)
        with torch.no_grad():
            _ = model(incoming_clip, stream=True, stream_id="stream_0")

        frames_processed += clip_len

        if frames_processed % sample_interval < clip_len or step == num_steps:
            rss_mb = get_process_memory_mb()
            pytorch_alloc_mb = (
                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
            sampled_points.append(
                {
                    "frame": frames_processed,
                    "process_rss_mb": rss_mb,
                    "pytorch_allocated_mb": round(pytorch_alloc_mb, 2),
                    "delta_mb": round(rss_mb - initial_rss_mb, 2),
                }
            )

        if step % (num_steps // 10 or 1) == 0 or step == num_steps:
            elapsed = time.perf_counter() - start_time
            fps = frames_processed / max(elapsed, 1e-4)
            current_rss = sampled_points[-1]["process_rss_mb"] if sampled_points else initial_rss_mb
            print(
                f"  [{step:5d}/{num_steps}] Frames: {frames_processed:6d} | "
                f"RSS: {current_rss:.1f} MB | Speed: {fps:.1f} FPS"
            )

    total_time = time.perf_counter() - start_time
    avg_fps = frames_processed / max(total_time, 1e-4)

    # Compute Memory Statistics
    rss_values = [p["process_rss_mb"] for p in sampled_points]
    peak_rss = max(rss_values) if rss_values else initial_rss_mb
    min_rss = min(rss_values) if rss_values else initial_rss_mb
    final_rss = rss_values[-1] if rss_values else initial_rss_mb
    rss_drift = final_rss - initial_rss_mb

    results = {
        "status": "SUCCESS",
        "total_frames": frames_processed,
        "sampling_interval_frames": sample_interval,
        "device": device.type,
        "total_time_seconds": round(total_time, 2),
        "average_fps": round(avg_fps, 1),
        "initial_rss_mb": initial_rss_mb,
        "peak_rss_mb": peak_rss,
        "min_rss_mb": min_rss,
        "final_rss_mb": final_rss,
        "net_drift_mb": round(rss_drift, 2),
        "is_bounded_O1": abs(rss_drift) < 50.0,  # Strict bounding check (<50MB fluctuation)
        "samples": sampled_points,
    }

    print("\n--- Empirical O(1) Memory Verification ---")
    print(f"Total Frames Processed: {frames_processed:,}")
    print(f"Initial Memory:         {initial_rss_mb:.2f} MB")
    print(f"Peak Memory:            {peak_rss:.2f} MB")
    print(f"Final Memory:           {final_rss:.2f} MB")
    print(f"Net Memory Drift:       {rss_drift:+.2f} MB")
    print(f"Memory Complexity Claim: {'VERIFIED O(1) BOUNDED' if results['is_bounded_O1'] else 'UNBOUNDED'}")
    print("=" * 65)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Memory profile JSON saved to: {output_json}")

    # Generate publication plot if matplotlib is available
    if output_plot and len(sampled_points) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_plot.parent.mkdir(parents=True, exist_ok=True)
            frames_x = [p["frame"] for p in sampled_points]
            rss_y = [p["process_rss_mb"] for p in sampled_points]

            plt.figure(figsize=(8, 4.5), dpi=300)
            plt.plot(frames_x, rss_y, color="#1f77b4", linewidth=2.0, label="Process RSS RAM (MB)")
            plt.axhline(y=initial_rss_mb, color="#d62728", linestyle="--", alpha=0.7, label=f"Initial Baseline ({initial_rss_mb:.1f} MB)")
            plt.title("Empirical Streaming Memory Invariance: Bounded $\\mathcal{O}(1)$ FIFO Verification", fontsize=12, pad=12)
            plt.xlabel("Stream Duration (Frames Ingested)", fontsize=10)
            plt.ylabel("Memory Footprint (MB)", fontsize=10)
            plt.ylim(bottom=max(0, min_rss - 20), top=peak_rss + 30)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(output_plot)
            plt.close()
            print(f"[OK] Memory plot saved to: {output_plot}")
        except Exception as e:
            print(f"[WARNING] Could not generate plot: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Empirically verify O(1) streaming memory.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", "--total-frames", type=int, dest="frames", default=10000)
    parser.add_argument("--interval", "--sample-interval", type=int, dest="interval", default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", "--output", "-o", type=Path, dest="output_json", default=PROJECT_ROOT / "outputs" / "profiling" / "memory_profile.json")
    parser.add_argument("--output-plot", type=Path, default=PROJECT_ROOT / "figures" / "memory_drift_profile.png")
    args = parser.parse_args()

    profile_memory_stream(
        checkpoint_path=args.checkpoint,
        total_frames=args.frames,
        sample_interval=args.interval,
        device=args.device,
        output_json=args.output_json,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
