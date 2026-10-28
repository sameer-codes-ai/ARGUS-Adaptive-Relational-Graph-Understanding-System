"""
Single-video / clip inference and explainability script for ARGUS.
Runs end-to-end anomaly & aggression detection on an input video or frame folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
import cv2
import numpy as np

from argus import ARGUS, AlertDispatcher, generate_dashboard_html
from argus.config import ARGUSConfig
from argus.data import load_clip_from_path
from argus.checkpoint import load_checkpoint, model_state_dict, validate_component_metadata


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
    parser = argparse.ArgumentParser(description="Run ARGUS inference on a video or frame directory.")
    parser.add_argument('--input', type=str, required=True, help='Path to video file (.mp4/.avi) or frame folder.')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/checkpoints/best.pt'), help='Path to model checkpoint.')
    parser.add_argument('--frames', type=int, default=16, help='Number of input frames to sample.')
    parser.add_argument('--image-size', type=int, default=None, help='Spatial resolution. Auto-detected from checkpoint if not specified.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly classification decision threshold.')
    parser.add_argument('--output-json', type=Path, default=None, help='Path to save inference result JSON.')
    parser.add_argument('--dashboard', type=Path, default=None, help='Optional path to export interactive HTML dashboard.')
    parser.add_argument('--stream', action='store_true', help='Update temporal streaming memory.')
    parser.add_argument('--stream-id', type=str, default='default', help='Stable source identity for streaming memory.')
    parser.add_argument('--allow-random-model', action='store_true', help='Explicitly allow untrained random weights for development.')
    parser.add_argument('--allow-unsafe-checkpoint', action='store_true', help='Allow a trusted legacy pickle checkpoint.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Inference Device: {device.upper()}")

    overall_start = time.perf_counter()

    # Step 1: Model loading
    t0 = time.perf_counter()
    if args.checkpoint.exists():
        state = load_checkpoint(args.checkpoint, map_location=device, allow_unsafe=args.allow_unsafe_checkpoint)
        saved_cfg = state.get('config', {})
        config = ARGUSConfig.from_dict(saved_cfg, strict=True) if saved_cfg else ARGUSConfig().validate()
        if args.image_size is not None:
            config.image_size = args.image_size
        model = ARGUS(config=config).to(device)
        validate_component_metadata(state, model)
        model.load_state_dict(model_state_dict(state))
        load_time = time.perf_counter() - t0
        print(f"[OK] [1/3] Loaded model checkpoint: {args.checkpoint} (Epoch {state.get('epoch', '?') if isinstance(state, dict) else '?'}) in {format_duration(load_time)}")
    elif args.allow_random_model:
        print(f"DEVELOPMENT MODE: checkpoint {args.checkpoint} not found; using untrained random weights.")
        config = ARGUSConfig(image_size=args.image_size or 96)
        model = ARGUS(config=config).to(device)
        load_time = time.perf_counter() - t0
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Use --allow-random-model only for explicit development tests."
        )

    model.eval()

    # Step 2: Video reading & decoding
    t1 = time.perf_counter()
    print(f"[*] [2/3] Reading & preprocessing video clip from {input_path.name}...")
    clip = load_clip_from_path(input_path, target_frames=args.frames, image_size=config.image_size)
    clip_batch = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
    decode_time = time.perf_counter() - t1
    print(f"[OK] [2/3] Decoded {args.frames} frames ({config.image_size}x{config.image_size}) in {format_duration(decode_time)}")

    # Step 3: Neural reasoning & graph inference
    t2 = time.perf_counter()
    print(f"[*] [3/3] Running ARGUS adaptive sampling + scene graph reasoning...")
    with torch.no_grad():
        out = model(clip_batch, stream=args.stream, stream_id=args.stream_id)
    infer_time = time.perf_counter() - t2
    print(f"[OK] [3/3] Neural inference finished in {format_duration(infer_time)}")

    total_pipeline_time = time.perf_counter() - overall_start

    valid_length = int(out.sequence_lengths[0].item())
    scores_seq = out.scores[0, :valid_length].cpu().numpy().tolist()
    max_score = float(np.max(scores_seq))
    mean_score = float(np.mean(scores_seq))
    is_anomaly = bool(max_score >= args.threshold)
    sampled_indices = out.sampled_indices[0].cpu().tolist() if isinstance(out.sampled_indices[0], torch.Tensor) else list(out.sampled_indices[0])

    print("\n" + "=" * 58)
    print("                 ARGUS INFERENCE RESULT                 ")
    print("=" * 58)
    status_str = "ANOMALOUS / AGGRESSIVE" if is_anomaly else "NORMAL / NON-VIOLENT"
    print(f"  Classification:      [{status_str}]")
    print(f"  Peak Anomaly Score:  {max_score:.4f} (Threshold: {args.threshold:.2f})")
    print(f"  Mean Anomaly Score:  {mean_score:.4f}")
    print(f"  Sampled Keyframes:   {sampled_indices} (Total: {len(sampled_indices)} / {args.frames})")
    print("-" * 58)
    print("  Latency Breakdown:")
    print(f"    - Frame Decode/Prep:  {format_duration(decode_time)}")
    print(f"    - Neural Inference:   {format_duration(infer_time)} ({args.frames / max(infer_time, 1e-6):.1f} FPS)")
    print(f"    - Total Execution:    {format_duration(total_pipeline_time)}")
    print("-" * 58)
    print("  Keyframe Timeline Scores:")
    for idx, (kf, sc) in enumerate(zip(sampled_indices, scores_seq)):
        bar = "#" * int(sc * 25)
        print(f"    Frame {kf:03d} -> Score: {sc:.4f} [{bar:<25}]")
    print("=" * 58)

    dispatcher = AlertDispatcher(threshold=args.threshold)
    event = dispatcher.dispatch(
        video_source=str(input_path),
        timeline_scores=scores_seq,
        keyframe_indices=sampled_indices,
        boxes_per_frame=out.boxes[0] if out.boxes else None,
        semantic_types_per_frame=out.semantic_types[0] if getattr(out, 'semantic_types', None) else None,
        detection_scores_per_frame=out.detection_scores[0] if getattr(out, 'detection_scores', None) else None,
        edge_indices_per_frame=out.edge_indices[0] if getattr(out, 'edge_indices', None) else None,
        edge_attrs_per_frame=out.edge_attrs[0] if getattr(out, 'edge_attrs', None) else None,
        detector_impl=getattr(out, 'detector_impl', 'yolo'),
        class_ids_per_frame=out.class_ids[0] if getattr(out, 'class_ids', None) else None,
    )

    result = {
        'input_path': str(input_path.resolve()),
        'is_anomaly': is_anomaly,
        'peak_anomaly_score': max_score,
        'mean_anomaly_score': mean_score,
        'timeline_scores': scores_seq,
        'sampled_keyframes': sampled_indices,
        'detector_implementation': getattr(out, 'detector_impl', 'yolo'),
        'person_count': event.person_count if event else 0,
        'vehicle_count': event.vehicle_count if event else 0,
        'bag_count': event.bag_count if event else 0,
        'weapon_count': event.weapon_count if event else 0,
        'other_count': event.other_count if event else 0,
        'object_counts': event.object_counts if event else {},
        'person_weapon_edges': event.person_weapon_edges if event else 0,
        'num_person_interactions': event.num_person_interactions if event else 0,
        'summary': event.summary if event else '',
        'timings': {
            'decode_s': decode_time,
            'inference_s': infer_time,
            'total_s': total_pipeline_time,
        }
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Saved inference results to {args.output_json}")

    if args.dashboard and event:
        fps_rate = float(args.frames / max(infer_time, 1e-6))
        lat_ms = float((infer_time / max(args.frames, 1)) * 1000.0)
        generate_dashboard_html(event, args.dashboard, fps=fps_rate, latency_ms=lat_ms)
        print(f"[OK] Generated interactive dashboard: {args.dashboard}")


if __name__ == '__main__':
    main()


