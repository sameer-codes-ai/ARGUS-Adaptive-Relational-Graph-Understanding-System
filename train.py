import argparse
import io
import json
import os
from pathlib import Path
import shutil
import time
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from argus import ARGUS
from argus.config import ARGUSConfig
from argus.data import VideoManifest, SyntheticSurveillance
from argus.losses import anomaly_loss
from argus.metrics import binary_metrics


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string (e.g., '1h 23m 45s' or '12.4s')."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def safe_torch_save(obj, target_path: Path, max_retries: int = 5, retry_delay: float = 0.5):
    """
    Safely save a PyTorch object to disk on Windows.
    Uses an in-memory buffer / temporary file with atomic replacement to prevent
    Windows error code 1224 (ERROR_USER_MAPPED_FILE / locked file handle).
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.stem}_tmp_{os.getpid()}_{int(time.time() * 1000)}.pt")

    try:
        torch.save(obj, temp_path)
        for attempt in range(max_retries):
            try:
                os.replace(temp_path, target_path)
                return
            except OSError:
                if attempt == max_retries - 1:
                    try:
                        shutil.copy2(temp_path, target_path)
                        if temp_path.exists():
                            temp_path.unlink()
                        return
                    except Exception:
                        torch.save(obj, target_path)
                        if temp_path.exists():
                            temp_path.unlink()
                        return
                time.sleep(retry_delay)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARGUS Anomaly & Aggression Detection Model.")
    parser.add_argument('--config', type=Path, default=Path('configs/train.yaml'), help='Path to YAML config file.')
    parser.add_argument('--manifest', type=str, help='Path to training manifest CSV.')
    parser.add_argument('--val-manifest', type=str, help='Path to validation manifest CSV.')
    parser.add_argument('--resume', '--checkpoint', dest='resume', type=Path, default=None, help='Path to checkpoint (.pt) to resume or fine-tune from.')
    parser.add_argument('--pretrained', type=Path, default=None, help='Path to an existing checkpoint (e.g. baseline inner layer) to initialize matching weights from.')
    parser.add_argument('--reset-optimizer', action='store_true', help='Reset optimizer and scheduler when resuming for fresh fine-tuning.')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for fast smoke testing.')
    parser.add_argument('--frames', type=int, help='Number of frames per clip.')
    parser.add_argument('--image-size', type=int, help='Spatial image resolution (height=width).')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, help='Batch size for training.')
    parser.add_argument('--lr', type=float, help='Learning rate.')
    parser.add_argument('--backbone-lr', type=float, help='Learning rate for backbone (default: 0.1x of base lr).')
    parser.add_argument('--weight-decay', type=float, help='Weight decay for AdamW.')
    parser.add_argument('--mil-weight', type=float, help='Weight for Multiple Instance Learning (MIL) ranking loss.')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentations.')
    parser.add_argument('--unfreeze-backbone', action='store_true', help='Unfreeze spatial ConvNeXt backbone for joint end-to-end training of both layers.')
    parser.add_argument('--start-epoch', type=int, default=None, help='Explicit starting epoch number (overrides checkpoint epoch).')
    parser.add_argument('--reset-epoch', action='store_true', help='Reset starting epoch number to 1 even when resuming weights.')
    parser.add_argument('--output-dir', type=Path, help='Output directory for checkpoints.')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches per epoch (for quick debug).')
    return parser.parse_args()


def load_config(args) -> dict:
    config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    # CLI overrides
    manifest = args.manifest or data_cfg.get('train_manifest')
    val_manifest = args.val_manifest or data_cfg.get('val_manifest')
    frames = args.frames or data_cfg.get('frames', 16)
    image_size = args.image_size or model_cfg.get('image_size', 96)

    epochs = args.epochs or train_cfg.get('epochs', 10)
    batch_size = args.batch_size or train_cfg.get('batch_size', 4)
    lr = args.lr or train_cfg.get('lr', 1e-4)
    backbone_lr = args.backbone_lr or train_cfg.get('backbone_lr', None)
    weight_decay = args.weight_decay or train_cfg.get('weight_decay', 1e-4)
    smoothness_weight = train_cfg.get('smoothness_weight', 0.1)
    sparsity_weight = train_cfg.get('sparsity_weight', 0.01)
    mil_weight = args.mil_weight if args.mil_weight is not None else train_cfg.get('mil_weight', 0.5)
    augment = not args.no_augment
    unfreeze_backbone = args.unfreeze_backbone or train_cfg.get('unfreeze_backbone', False)
    start_epoch = args.start_epoch if args.start_epoch is not None else train_cfg.get('start_epoch', None)
    reset_epoch = args.reset_epoch or train_cfg.get('reset_epoch', False)
    output_dir = args.output_dir or Path(train_cfg.get('output_dir', 'outputs/checkpoints'))

    return {
        'manifest': manifest,
        'val_manifest': val_manifest,
        'resume': args.resume,
        'pretrained': args.pretrained,
        'reset_optimizer': args.reset_optimizer,
        'synthetic': args.synthetic,
        'frames': frames,
        'image_size': image_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'backbone_lr': backbone_lr,
        'weight_decay': weight_decay,
        'smoothness_weight': smoothness_weight,
        'sparsity_weight': sparsity_weight,
        'mil_weight': mil_weight,
        'augment': augment,
        'unfreeze_backbone': unfreeze_backbone,
        'start_epoch': start_epoch,
        'reset_epoch': reset_epoch,
        'output_dir': output_dir,
        'max_batches': args.max_batches,
        'model_cfg': model_cfg,
    }


def create_loader(manifest_path, synthetic, frames, image_size, batch_size, shuffle=True, augment=False):
    if synthetic:
        ds = SyntheticSurveillance(count=48, frames=frames, size=image_size, augment=augment)
    elif manifest_path:
        ds = VideoManifest(manifest_path, frames=frames, image_size=image_size, augment=augment)
    else:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(model, dataloader, optimizer, device, smoothness_w, sparsity_w, mil_w, epoch_idx, total_epochs, max_batches=None, scaler=None, use_amp=False):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    batch_count = 0

    total_len = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    pbar = tqdm(
        total=total_len,
        desc=f"Epoch {epoch_idx:02d}/{total_epochs:02d} [Train]",
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar:25}{r_bar}"
    )

    epoch_start = time.perf_counter()

    for frames, y in dataloader:
        frames, y = frames.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            out = model(frames)
        loss, loss_dict = anomaly_loss(
            out.scores.float(), y,
            smoothness_weight=smoothness_w,
            sparsity_weight=sparsity_w,
            mil_weight=mil_w
        )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = out.scores.max(dim=1).values.detach()
        all_preds.append(preds)
        all_labels.append(y.detach())

        batch_count += 1
        current_avg_loss = total_loss / batch_count

        pred_bin = (preds >= 0.5).long()
        batch_acc = (pred_bin == y.long()).float().mean().item()

        postfix_dict = {
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{current_avg_loss:.4f}",
            'acc': f"{batch_acc * 100:.1f}%",
        }
        if loss_dict.get('mil', 0.0) > 0.0:
            postfix_dict['mil'] = f"{loss_dict['mil']:.3f}"

        pbar.set_postfix(postfix_dict)
        pbar.update(1)

        if max_batches and batch_count >= max_batches:
            break

    pbar.close()
    elapsed = time.perf_counter() - epoch_start
    avg_loss = total_loss / max(batch_count, 1)
    metrics = binary_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['epoch_time_s'] = elapsed
    return avg_loss, metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, smoothness_w, sparsity_w, mil_w, epoch_idx, total_epochs, max_batches=None, use_amp=False):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    batch_count = 0

    total_len = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    pbar = tqdm(
        total=total_len,
        desc=f"Epoch {epoch_idx:02d}/{total_epochs:02d} [Val]  ",
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar:25}{r_bar}"
    )

    eval_start = time.perf_counter()

    for frames, y in dataloader:
        frames, y = frames.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            out = model(frames)
        loss, _ = anomaly_loss(
            out.scores.float(), y,
            smoothness_weight=smoothness_w,
            sparsity_weight=sparsity_w,
            mil_weight=mil_w
        )

        total_loss += loss.item()
        preds = out.scores.max(dim=1).values.detach()
        all_preds.append(preds)
        all_labels.append(y.detach())

        batch_count += 1
        current_avg_loss = total_loss / batch_count

        pbar.set_postfix({
            'val_loss': f"{current_avg_loss:.4f}",
        })
        pbar.update(1)

        if max_batches and batch_count >= max_batches:
            break

    pbar.close()
    elapsed = time.perf_counter() - eval_start
    avg_loss = total_loss / max(batch_count, 1)
    metrics = binary_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['val_time_s'] = elapsed
    return avg_loss, metrics


def main():
    args = parse_args()
    cfg = load_config(args)

    if not cfg['synthetic'] and not cfg['manifest']:
        raise ValueError("Either --synthetic or a valid --manifest / config train_manifest is required.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device.upper()}")

    start_epoch = 1
    best_score = -1.0
    checkpoint_data = None

    if cfg['resume']:
        checkpoint_path = Path(cfg['resume'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from not found: {checkpoint_path}")
        print(f"[*] Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint_bytes = io.BytesIO(f.read())
        checkpoint_data = torch.load(checkpoint_bytes, map_location=device, weights_only=False)

        # Restore architecture config directly from checkpoint if available
        if isinstance(checkpoint_data, dict) and 'config' in checkpoint_data and checkpoint_data['config']:
            model_config = ARGUSConfig.from_dict(checkpoint_data['config'])
            # If CLI image_size was explicitly passed, override resolution
            if args.image_size:
                model_config.image_size = args.image_size
            else:
                cfg['image_size'] = model_config.image_size
            print(f"  [*] Using architecture configuration from checkpoint: dim={model_config.dim}, layers={model_config.temporal_layers}, img_size={model_config.image_size}")
        else:
            model_config = ARGUSConfig.from_dict(cfg['model_cfg'])
            model_config.image_size = cfg['image_size']
    else:
        # Build model from config file
        model_config = ARGUSConfig.from_dict(cfg['model_cfg'])
        model_config.image_size = cfg['image_size']

    model = ARGUS(config=model_config).to(device)

    if cfg.get('pretrained') and Path(cfg['pretrained']).exists():
        pre_path = Path(cfg['pretrained'])
        print(f"Loading pre-trained weights from: {pre_path}")
        pre_data = torch.load(pre_path, map_location='cpu', weights_only=False)
        pre_state = pre_data['model'] if (isinstance(pre_data, dict) and 'model' in pre_data) else pre_data
        model_state = model.state_dict()
        matching = {k: v for k, v in pre_state.items() if k in model_state and model_state[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(matching, strict=False)
        print(f"  [OK] Initialized {len(matching)}/{len(model_state)} matching parameter tensors from {pre_path.name}.")

    if cfg['resume'] and checkpoint_data is not None:
        if isinstance(checkpoint_data, dict) and 'model' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model'])
            ckpt_epoch = checkpoint_data.get('epoch', 0)
            if cfg['reset_epoch']:
                start_epoch = 1
            elif cfg['start_epoch'] is not None:
                start_epoch = cfg['start_epoch']
            else:
                start_epoch = ckpt_epoch + 1
            prev_metrics = checkpoint_data.get('val_metrics') or checkpoint_data.get('train_metrics')
            if prev_metrics and 'f1' in prev_metrics:
                best_score = prev_metrics['f1']
            print(f"  [OK] Successfully loaded weights (Checkpoint Epoch: {ckpt_epoch} | Resuming as Epoch {start_epoch} | Best F1: {best_score:.4f})")
        else:
            model.load_state_dict(checkpoint_data)
            print(f"  [OK] Successfully loaded model state dictionary.")

    if cfg['unfreeze_backbone'] and model.backbone is not None:
        model.backbone.unfreeze()
        model.cfg.backbone_frozen = False
        print("  [*] Unfroze spatial ConvNeXt backbone for joint end-to-end training.")

    backbone_lr = cfg.get('backbone_lr') if cfg.get('backbone_lr') is not None else (cfg['lr'] * 0.1)
    if model.backbone is not None and not model.cfg.backbone_frozen:
        backbone_param_ids = set(id(p) for p in model.backbone.model.parameters())
        backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids and p.requires_grad]
        head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids and p.requires_grad]
        param_groups = [
            {'params': head_params, 'lr': cfg['lr']},
            {'params': backbone_params, 'lr': backbone_lr}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])
        print(f"  [*] Differential LR configured: Heads LR = {cfg['lr']}, Backbone LR = {backbone_lr}")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler(enabled=True) if use_amp else None

    if cfg['resume'] and not cfg['reset_optimizer']:
        if isinstance(checkpoint_data, dict) and 'optimizer' in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer'])
                print("  [OK] Restored optimizer state from checkpoint.")
            except Exception as e:
                print(f"  [!] Note: Reinitialized optimizer ({e}).")
        if isinstance(checkpoint_data, dict) and 'scheduler' in checkpoint_data:
            try:
                scheduler.load_state_dict(checkpoint_data['scheduler'])
                print("  [OK] Restored scheduler state from checkpoint.")
            except Exception as e:
                print(f"  [!] Note: Reinitialized scheduler ({e}).")

    train_loader = create_loader(
        cfg['manifest'], cfg['synthetic'], cfg['frames'], cfg['image_size'],
        cfg['batch_size'], shuffle=True, augment=cfg['augment']
    )
    val_loader = create_loader(
        cfg['val_manifest'], cfg['synthetic'], cfg['frames'], cfg['image_size'],
        cfg['batch_size'], shuffle=False, augment=False
    ) if (cfg['val_manifest'] or cfg['synthetic']) else None

    total_epochs = start_epoch + cfg['epochs'] - 1

    print(f"Dataset: {len(train_loader.dataset)} train clips", end="")
    if val_loader:
        print(f" | {len(val_loader.dataset)} val clips")
    else:
        print()
    print(f"Training plan: Epochs {start_epoch} to {total_epochs} (+{cfg['epochs']} epochs) | batch size {cfg['batch_size']} | {cfg['frames']} frames/clip | {cfg['image_size']}x{cfg['image_size']} res | MIL weight: {cfg['mil_weight']} | Augment: {cfg['augment']}\n")

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    history_path = output_dir / 'training_history.json'
    if cfg['resume'] and history_path.exists() and not cfg['reset_epoch']:
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            # Prune any previous run entries that overlap with or exceed start_epoch
            history = [entry for entry in history if entry.get('epoch', 0) < start_epoch]
            print(f"  [OK] Loaded {len(history)} previous valid epoch(s) from training history.")
        except Exception:
            history = []
    elif cfg['resume'] and isinstance(checkpoint_data, dict) and 'epoch' in checkpoint_data and not cfg['reset_epoch']:
        history = [{
            'epoch': checkpoint_data['epoch'],
            'train_metrics': checkpoint_data.get('train_metrics'),
            'val_metrics': checkpoint_data.get('val_metrics'),
        }]
        print(f"  [OK] Initialized history from checkpoint metadata (Epoch {checkpoint_data['epoch']}).")
    training_start_time = time.perf_counter()
    epoch_durations = []

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            cfg['smoothness_weight'], cfg['sparsity_weight'], cfg['mil_weight'],
            epoch, total_epochs, cfg['max_batches'], scaler=scaler, use_amp=use_amp
        )
        scheduler.step()

        val_metrics = None
        val_loss = 0.0
        if val_loader:
            val_loss, val_metrics = evaluate_epoch(
                model, val_loader, device,
                cfg['smoothness_weight'], cfg['sparsity_weight'], cfg['mil_weight'],
                epoch, total_epochs, cfg['max_batches'], use_amp=use_amp
            )

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = total_epochs - epoch
        estimated_remaining_time = avg_epoch_duration * remaining_epochs
        total_elapsed = time.perf_counter() - training_start_time

        status_line = (
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']*100:.1f}% | F1: {train_metrics['f1']:.3f} | AUC: {train_metrics['auc_roc']:.3f}"
        )
        if val_metrics:
            status_line += (
                f" || Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.1f}% | Val F1: {val_metrics['f1']:.3f} | Val AUC: {val_metrics['auc_roc']:.3f}"
            )
        status_line += f"  [{format_duration(epoch_duration)} / ETA: {format_duration(estimated_remaining_time)}]"
        print(status_line)

        current_score = val_metrics['f1'] if val_metrics and not (val_metrics['f1'] != val_metrics['f1']) else train_metrics['f1']
        is_best = current_score >= best_score

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': model_config.to_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        safe_torch_save(checkpoint, output_dir / 'latest.pt')
        if is_best:
            best_score = current_score
            safe_torch_save(checkpoint, output_dir / 'best.pt')
            print(f"  [*] New best checkpoint saved! (Score: {best_score:.4f})")

        new_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'epoch_duration_s': epoch_duration,
        }
        existing_idx = next((i for i, h in enumerate(history) if h.get('epoch') == epoch), None)
        if existing_idx is not None:
            history[existing_idx] = new_entry
        else:
            history.append(new_entry)
        # Persist history atomically (write temp, then rename) to prevent corruption on crash
        history_tmp = output_dir / f'training_history_tmp_{os.getpid()}.json'
        with open(history_tmp, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        try:
            os.replace(history_tmp, output_dir / 'training_history.json')
        except OSError:
            shutil.copy2(history_tmp, output_dir / 'training_history.json')
            if history_tmp.exists():
                history_tmp.unlink()

    total_training_duration = time.perf_counter() - training_start_time
    print(f"\n[OK] Training completed in {format_duration(total_training_duration)}! Artifacts saved to {output_dir}")


if __name__ == '__main__':
    main()
