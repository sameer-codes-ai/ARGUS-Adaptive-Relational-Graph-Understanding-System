import argparse
import io
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from argus import ARGUS
from argus.config import ARGUSConfig
from argus.data import VideoManifest, SyntheticSurveillance
from argus.metrics import binary_metrics


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARGUS Model Checkpoint.")
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint (.pt).')
    parser.add_argument('--manifest', type=str, help='Path to evaluation manifest CSV.')
    parser.add_argument('--synthetic', action='store_true', help='Evaluate on synthetic data.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation.')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames per clip.')
    parser.add_argument('--image-size', type=int, default=96, help='Spatial resolution.')
    parser.add_argument('--output-json', type=Path, default=None, help='Optional path to save metrics JSON.')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit evaluation samples for fast testing.')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.synthetic and not args.manifest:
        raise ValueError("Either --synthetic or --manifest must be specified.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation compute device: {device.upper()}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    with open(args.checkpoint, 'rb') as f:
        checkpoint_bytes = io.BytesIO(f.read())
    state = torch.load(checkpoint_bytes, map_location=device, weights_only=False)

    saved_cfg = state.get('config', {})
    config = ARGUSConfig.from_dict(saved_cfg) if saved_cfg else ARGUSConfig()
    config.image_size = args.image_size

    model = ARGUS(config=config).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (Epoch {state.get('epoch', '?')})")

    if args.synthetic:
        dataset = SyntheticSurveillance(count=48, frames=args.frames, size=args.image_size)
    else:
        dataset = VideoManifest(args.manifest, frames=args.frames, image_size=args.image_size)

    total_samples_to_eval = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {total_samples_to_eval} samples...\n")

    all_scores, all_labels = [], []
    total_time = 0.0
    total_frames = 0
    samples_processed = 0

    eval_start_time = time.perf_counter()
    pbar = tqdm(
        total=total_samples_to_eval,
        desc="Evaluating clips",
        unit="clip",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:25}{r_bar}"
    )

    with torch.no_grad():
        for frames, labels in loader:
            b, t = frames.shape[:2]
            frames = frames.to(device)

            step_start = time.perf_counter()
            out = model(frames)
            step_elapsed = time.perf_counter() - step_start

            total_time += step_elapsed
            total_frames += b * t

            clip_scores = out.scores.max(dim=1).values.cpu()
            all_scores.append(clip_scores)
            all_labels.append(labels.cpu())

            samples_processed += b
            current_fps = (total_frames / max(total_time, 1e-6))
            pbar.set_postfix({
                'fps': f"{current_fps:.1f}",
                'ms/clip': f"{(total_time / max(samples_processed, 1))*1000:.1f}ms",
            })
            pbar.update(b)

            if args.max_samples and samples_processed >= args.max_samples:
                break

    pbar.close()
    wall_clock_time = time.perf_counter() - eval_start_time

    scores_t = torch.cat(all_scores)
    labels_t = torch.cat(all_labels)

    metrics = binary_metrics(scores_t, labels_t)
    fps = total_frames / max(total_time, 1e-6)
    ms_per_clip = (total_time / max(samples_processed, 1)) * 1000.0

    metrics['fps'] = round(fps, 2)
    metrics['latency_ms_per_clip'] = round(ms_per_clip, 2)
    metrics['samples_evaluated'] = samples_processed
    metrics['total_time_s'] = round(wall_clock_time, 2)

    pred_binary = (scores_t >= 0.5).long()
    labels_long = labels_t.long()
    tp = ((pred_binary == 1) & (labels_long == 1)).sum().item()
    fp = ((pred_binary == 1) & (labels_long == 0)).sum().item()
    fn = ((pred_binary == 0) & (labels_long == 1)).sum().item()
    tn = ((pred_binary == 0) & (labels_long == 0)).sum().item()

    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['tn'] = tn
    metrics['fn'] = fn

    print("\n" + "=" * 55)
    print("              ARGUS EVALUATION RESULTS              ")
    print("=" * 55)
    print(f"  ROC-AUC:            {metrics['auc_roc']:.4f}")
    print(f"  Average Precision:  {metrics['average_precision']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1']:.4f}")
    print("-" * 55)
    print(f"  Confusion Matrix:   TP={tp:d} | FP={fp:d} | TN={tn:d} | FN={fn:d}")
    print(f"  Processing Speed:   {fps:.1f} FPS ({ms_per_clip:.1f} ms/clip)")
    print(f"  Total Time:         {format_duration(wall_clock_time)} ({samples_processed} clips)")
    print("=" * 55)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[OK] Saved metrics to {args.output_json}")


if __name__ == '__main__':
    main()
