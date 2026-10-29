import argparse
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_DEPS = PROJECT_ROOT / ".testdeps"
if TEST_DEPS.is_dir() and str(TEST_DEPS) not in sys.path:
    sys.path.insert(0, str(TEST_DEPS))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from argus import ARGUS
from argus.config import ARGUSConfig
from argus.data import VideoManifest, SyntheticSurveillance
from argus.losses import anomaly_loss, compute_model_loss
from argus.baselines import build_model
from argus.metrics import binary_metrics
from argus.checkpoint import (
    load_checkpoint,
    load_partial_checkpoint,
    model_state_dict,
    validate_component_metadata,
)


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
    parser.add_argument('--model-type', type=str, choices=['argus', 'convnext_mil', 'causal_tsm', 'rtfm_lite'], default='argus', help='Model architecture to train.')
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
    parser.add_argument('--workers', type=int, default=None, help='Data loader worker processes.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible training.')
    parser.add_argument('--allow-unsafe-checkpoint', action='store_true', help='Allow trusted legacy pickle checkpoints that fail safe loading.')
    parser.add_argument('--eco-mode', action='store_true', help='Multi-tasking eco-mode: limits CPU threads, sets BelowNormal priority, and caps VRAM.')
    parser.add_argument('--cpu-threads', type=int, default=None, help='Limit PyTorch CPU threads (e.g. 4 on Ryzen 7 7840HS).')
    parser.add_argument('--max-vram-fraction', type=float, default=None, help='Cap PyTorch CUDA VRAM allocation fraction (e.g. 0.65 for 6GB RTX 3050).')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Compute device override (cpu or cuda).')
    return parser.parse_args()


def load_config(args) -> dict:
    config = {}
    if args.config and not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    # CLI overrides
    model_type = args.model_type or train_cfg.get('model_type', 'argus')
    manifest = args.manifest or data_cfg.get('train_manifest')
    val_manifest = args.val_manifest or data_cfg.get('val_manifest')
    frames = args.frames if args.frames is not None else data_cfg.get('frames', 16)
    image_size = args.image_size if args.image_size is not None else model_cfg.get('image_size', 96)

    epochs = args.epochs if args.epochs is not None else train_cfg.get('epochs', 10)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get('batch_size', 4)
    lr = args.lr if args.lr is not None else train_cfg.get('lr', 1e-4)
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else train_cfg.get('backbone_lr', None)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get('weight_decay', 1e-4)
    smoothness_weight = train_cfg.get('smoothness_weight', 0.1)
    sparsity_weight = train_cfg.get('sparsity_weight', 0.01)
    mil_weight = args.mil_weight if args.mil_weight is not None else train_cfg.get('mil_weight', 0.5)
    augment = not args.no_augment
    unfreeze_backbone = args.unfreeze_backbone or train_cfg.get('unfreeze_backbone', False)
    start_epoch = args.start_epoch if args.start_epoch is not None else train_cfg.get('start_epoch', None)
    reset_epoch = args.reset_epoch or train_cfg.get('reset_epoch', False)
    output_dir = args.output_dir or Path(train_cfg.get('output_dir', 'outputs/checkpoints'))

    eco_mode = args.eco_mode or train_cfg.get('eco_mode', False)
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else train_cfg.get('cpu_threads', 4 if eco_mode else None)
    max_vram_fraction = args.max_vram_fraction if args.max_vram_fraction is not None else train_cfg.get('max_vram_fraction', 0.65 if eco_mode else None)

    return {
        'model_type': model_type,
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
        'workers': args.workers if args.workers is not None else train_cfg.get('workers', 0),
        'seed': args.seed if args.seed is not None else train_cfg.get('seed', 42),
        'allow_unsafe_checkpoint': args.allow_unsafe_checkpoint,
        'model_cfg': model_cfg,
        'eco_mode': eco_mode,
        'cpu_threads': cpu_threads,
        'max_vram_fraction': max_vram_fraction,
        'device': args.device or train_cfg.get('device', None),
    }


def create_loader(manifest_path, synthetic, frames, image_size, batch_size, shuffle=True, augment=False, workers=0, pin_memory=False):
    if synthetic:
        ds = SyntheticSurveillance(count=48, frames=frames, size=image_size, augment=augment)
    elif manifest_path:
        ds = VideoManifest(manifest_path, frames=frames, image_size=image_size, augment=augment)
    else:
        return None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def train_one_epoch(model, dataloader, optimizer, device, smoothness_w, sparsity_w, mil_w, epoch_idx, total_epochs, max_batches=None, scaler=None, use_amp=False, model_type='argus'):
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
        loss, loss_dict = compute_model_loss(
            model_type, out, y, model=model,
            smoothness_weight=smoothness_w,
            sparsity_weight=sparsity_w,
            mil_weight=mil_w,
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
        preds = out.scores.masked_fill(~out.valid_mask, float('-inf')).max(dim=1).values.detach()
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
def evaluate_epoch(model, dataloader, device, smoothness_w, sparsity_w, mil_w, epoch_idx, total_epochs, max_batches=None, use_amp=False, model_type='argus'):
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
        loss, _ = compute_model_loss(
            model_type, out, y, model=model,
            smoothness_weight=smoothness_w,
            sparsity_weight=sparsity_w,
            mil_weight=mil_w,
        )

        total_loss += loss.item()
        preds = out.scores.masked_fill(~out.valid_mask, float('-inf')).max(dim=1).values.detach()
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

    if cfg['epochs'] <= 0 or cfg['batch_size'] <= 0 or cfg['frames'] <= 0 or cfg['image_size'] <= 0:
        raise ValueError("epochs, batch_size, frames, and image_size must be positive")
    if cfg['lr'] < 0 or cfg['weight_decay'] < 0 or cfg['workers'] < 0:
        raise ValueError("learning rates, weight decay, and workers cannot be negative")
    if cfg['max_batches'] is not None and cfg['max_batches'] <= 0:
        raise ValueError("max_batches must be positive when provided")
    if cfg['resume'] and cfg['pretrained']:
        raise ValueError("Use either --resume or --pretrained, not both")

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    if cfg.get('eco_mode') or cfg.get('cpu_threads') or cfg.get('max_vram_fraction'):
        threads_to_use = cfg.get('cpu_threads') or 4
        torch.set_num_threads(threads_to_use)
        torch.set_num_interop_threads(max(1, threads_to_use // 2))

        priority_status = 'Default'
        try:
            import psutil
            p = psutil.Process()
            if sys.platform == 'win32':
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
            priority_status = 'BelowNormal (Desktop Responsive)'
        except Exception:
            pass

        vram_status = 'N/A'
        if torch.cuda.is_available() and cfg.get('max_vram_fraction'):
            try:
                frac = float(cfg['max_vram_fraction'])
                torch.cuda.set_per_process_memory_fraction(frac)
                vram_status = f'Capped at {frac*100:.0f}% (leaves ~{(1.0-frac)*100:.0f}% for OS/Browsers)'
            except Exception as e:
                vram_status = f'Error setting fraction ({e})'

        print(f'[ECO-MODE] Active | CPU Threads: {threads_to_use} | Priority: {priority_status} | VRAM: {vram_status}')

    if not cfg['synthetic'] and not cfg['manifest']:
        raise ValueError("Either --synthetic or a valid --manifest / config train_manifest is required.")

    requested_device = cfg.get('device')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA requested via --device cuda, but CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    elif requested_device in ('cpu', 'cuda'):
        device = requested_device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device.upper()}")

    start_epoch = 1
    best_score = -1.0
    best_loss = float("inf")
    checkpoint_data = None

    if cfg['resume']:
        checkpoint_path = Path(cfg['resume'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from not found: {checkpoint_path}")
        print(f"[*] Loading checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(
            checkpoint_path,
            map_location=device,
            allow_unsafe=cfg['allow_unsafe_checkpoint'],
        )

        # Restore architecture config directly from checkpoint if available
        if isinstance(checkpoint_data, dict) and 'config' in checkpoint_data and checkpoint_data['config']:
            model_config = ARGUSConfig.from_dict(checkpoint_data['config'], strict=True)
            # If CLI image_size was explicitly passed, override resolution
            if args.image_size:
                model_config.image_size = args.image_size
            else:
                cfg['image_size'] = model_config.image_size
            print(f"  [*] Using architecture configuration from checkpoint: dim={model_config.dim}, layers={model_config.temporal_layers}, img_size={model_config.image_size}")
        else:
            model_config = ARGUSConfig.from_dict(cfg['model_cfg'], strict=True)
            model_config.image_size = cfg['image_size']
    else:
        # Build model from config file
        model_config = ARGUSConfig.from_dict(cfg['model_cfg'], strict=True)
        model_config.image_size = cfg['image_size']

    model = build_model(model_type=cfg['model_type'], config=model_config).to(device)
    if checkpoint_data is not None:
        validate_component_metadata(checkpoint_data, model)

    if cfg.get('pretrained'):
        pre_path = Path(cfg['pretrained'])
        if not pre_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pre_path}")
        print(f"Loading pre-trained weights from: {pre_path}")
        pre_data = load_checkpoint(pre_path, map_location='cpu', allow_unsafe=cfg['allow_unsafe_checkpoint'])
        pre_state = model_state_dict(pre_data)
        model_state = model.state_dict()
        gat_mismatch = (
            "edge_gat.proj.weight" in pre_state
            and "edge_gat.proj.weight" in model_state
            and pre_state["edge_gat.proj.weight"].shape != model_state["edge_gat.proj.weight"].shape
        )
        if gat_mismatch:
            partial_info = load_partial_checkpoint(model, pre_data)
            print(
                f"  [OK] Partial schema initialization: loaded {len(partial_info['loaded'])}/{len(model_state)} "
                f"unchanged tensors from {pre_path.name} (initialized EdgeGAT fresh for {model.cfg.semantic_schema_version})."
            )
        else:
            matching = {k: v for k, v in pre_state.items() if k in model_state and model_state[k].shape == v.shape}
            model.load_state_dict(matching, strict=False)
            print(f"  [OK] Initialized {len(matching)}/{len(model_state)} matching parameter tensors from {pre_path.name}.")

    if cfg['resume'] and checkpoint_data is not None:
        if isinstance(checkpoint_data, dict) and 'model' in checkpoint_data:
            model.load_state_dict(model_state_dict(checkpoint_data))
            ckpt_epoch = checkpoint_data.get('epoch', 0)
            if cfg['reset_epoch']:
                start_epoch = 1
            elif cfg['start_epoch'] is not None:
                start_epoch = cfg['start_epoch']
            else:
                start_epoch = ckpt_epoch + 1
            prev_metrics = checkpoint_data.get('val_metrics') or checkpoint_data.get('train_metrics')
            if prev_metrics and 'auc_roc' in prev_metrics:
                best_score = checkpoint_data.get('best_score', prev_metrics['auc_roc'])
            elif prev_metrics and 'f1' in prev_metrics:
                best_score = checkpoint_data.get('best_score', prev_metrics['f1'])
            if 'val_loss' in checkpoint_data:
                best_loss = checkpoint_data.get('val_loss', float('inf'))
            print(f"  [OK] Successfully loaded weights (Checkpoint Epoch: {ckpt_epoch} | Resuming as Epoch {start_epoch} | Best Val AUC: {best_score:.4f})")
        else:
            model.load_state_dict(model_state_dict(checkpoint_data))
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
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

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
                scheduler.T_max = scheduler.last_epoch + cfg['epochs']
                print("  [OK] Restored scheduler state from checkpoint.")
            except Exception as e:
                print(f"  [!] Note: Reinitialized scheduler ({e}).")

    train_loader = create_loader(
        cfg['manifest'], cfg['synthetic'], cfg['frames'], cfg['image_size'],
        cfg['batch_size'], shuffle=True, augment=cfg['augment'], workers=cfg['workers'], pin_memory=use_amp
    )
    val_loader = create_loader(
        cfg['val_manifest'], cfg['synthetic'], cfg['frames'], cfg['image_size'],
        cfg['batch_size'], shuffle=False, augment=False, workers=cfg['workers'], pin_memory=use_amp
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
            epoch, total_epochs, cfg['max_batches'], scaler=scaler, use_amp=use_amp,
            model_type=cfg['model_type']
        )
        scheduler.step()

        val_metrics = None
        val_loss = 0.0
        if val_loader:
            val_loss, val_metrics = evaluate_epoch(
                model, val_loader, device,
                cfg['smoothness_weight'], cfg['sparsity_weight'], cfg['mil_weight'],
                epoch, total_epochs, cfg['max_batches'], use_amp=use_amp,
                model_type=cfg['model_type']
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

        # Phase 1.2: Standardize Checkpoint Selection strictly on Validation ROC-AUC
        if val_metrics and "auc_roc" in val_metrics and not np.isnan(val_metrics["auc_roc"]):
            current_score = float(val_metrics["auc_roc"])
            current_loss = float(val_loss)
        else:
            current_score = float(train_metrics["auc_roc"]) if "auc_roc" in train_metrics else 0.0
            current_loss = float(train_loss)

        # Strict improvement (> 1e-4) with validation loss as deterministic secondary tie-breaker
        if current_score > best_score + 1e-4:
            is_best = True
        elif abs(current_score - best_score) <= 1e-4 and current_loss < best_loss:
            is_best = True
        else:
            is_best = False

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': model_config.to_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_loss': val_loss,
            'model_type': cfg['model_type'],
            'best_score': max(best_score, current_score),
            'format_version': 2,
            'semantic_schema_version': getattr(model_config, 'semantic_schema_version', 'threat5'),
            'components': model.component_metadata(),
        }

        safe_torch_save(checkpoint, output_dir / 'latest.pt')
        if is_best:
            best_score = current_score
            best_loss = current_loss
            safe_torch_save(checkpoint, output_dir / 'best.pt')
            print(f"  [*] New best checkpoint saved! (Val AUC: {best_score:.4f} | Val Loss: {best_loss:.4f})")

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
