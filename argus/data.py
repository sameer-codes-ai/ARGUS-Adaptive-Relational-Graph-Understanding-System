from __future__ import annotations
import csv
from pathlib import Path
import re
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}


def natural_sort_key(s: str | Path):
    name = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]


def find_frame_files(dir_path: Path) -> list[Path]:
    frames_sub = dir_path / 'frames'
    search_dir = frames_sub if frames_sub.is_dir() else dir_path
    frame_files = [f for f in search_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    frame_files.sort(key=natural_sort_key)
    return frame_files


def load_clip_from_path(
    path: str | Path,
    target_frames: int = 16,
    image_size: int = 96
) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Media path not found: {path}")

    if path.is_file() and path.suffix.lower() == '.pt':
        clip = torch.load(path, weights_only=False)
        if isinstance(clip, dict):
            clip = clip.get('frames', clip.get('clip', list(clip.values())[0]))
        if not isinstance(clip, torch.Tensor):
            clip = torch.tensor(clip)
        if clip.ndim == 4 and clip.shape[1] in (1, 3):
            pass
        elif clip.ndim == 4 and clip.shape[-1] in (1, 3):
            clip = clip.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected tensor shape: {clip.shape}")
        clip = clip.float()
        if clip.max() > 1.0:
            clip = clip / 255.0

    elif path.is_dir():
        frame_files = find_frame_files(path)
        if not frame_files:
            raise ValueError(f"No image frames found in directory: {path}")

        total_avail = len(frame_files)
        if total_avail >= target_frames:
            sampled_ids = torch.linspace(0, total_avail - 1, target_frames).long().tolist()
        else:
            sampled_ids = list(range(total_avail))

        raw_frames = []
        for idx in sampled_ids:
            img_path = frame_files[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[0] != image_size or img.shape[1] != image_size:
                interp = cv2.INTER_AREA if img.shape[0] > image_size else cv2.INTER_LINEAR
                img = cv2.resize(img, (image_size, image_size), interpolation=interp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw_frames.append(torch.from_numpy(img).permute(2, 0, 1))

        if not raw_frames:
            raise ValueError(f"Failed to decode any frames from directory: {path}")

        clip = torch.stack(raw_frames).float() / 255.0

    elif path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and total_frames >= target_frames:
            target_indices = set(torch.linspace(0, total_frames - 1, target_frames).long().tolist())
        else:
            target_indices = None

        raw = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if target_indices is None or frame_idx in target_indices:
                if frame.shape[0] != image_size or frame.shape[1] != image_size:
                    interp = cv2.INTER_AREA if frame.shape[0] > image_size else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (image_size, image_size), interpolation=interp)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw.append(torch.from_numpy(rgb).permute(2, 0, 1))
            frame_idx += 1
            if target_indices and len(raw) >= target_frames:
                break
        cap.release()

        if not raw:
            raise ValueError(f"No frames could be read from video file: {path}")
        clip = torch.stack(raw).float() / 255.0

    else:
        raise ValueError(f"Unsupported path type: {path}")

    if clip.shape[0] < target_frames:
        repeat_count = target_frames - clip.shape[0]
        clip = torch.cat([clip, clip[-1:].expand(repeat_count, -1, -1, -1)], dim=0)
    elif clip.shape[0] > target_frames:
        ids = torch.linspace(0, clip.shape[0] - 1, target_frames).long()
        clip = clip[ids]

    if clip.shape[-2] != image_size or clip.shape[-1] != image_size:
        clip = F.interpolate(clip, size=(image_size, image_size), mode='bilinear', align_corners=False)

    return clip


class VideoManifest(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        frames: int = 16,
        image_size: int = 96
    ):
        self.manifest_path = Path(manifest_path)
        self.frames = frames
        self.image_size = image_size
        self.records: list[dict] = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get('path') or row.get('filepath') or row.get('video_path')
                lbl = row.get('label') or row.get('target') or row.get('anomaly', 0)
                if p:
                    self.records.append({
                        'path': p,
                        'label': float(lbl),
                        'dataset': row.get('dataset', 'unknown'),
                        'category': row.get('category', 'unknown'),
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        clip = load_clip_from_path(rec['path'], target_frames=self.frames, image_size=self.image_size)
        label = torch.tensor(rec['label'], dtype=torch.float32)
        return clip, label


class SyntheticSurveillance(Dataset):
    def __init__(self, count: int = 48, frames: int = 16, size: int = 96):
        self.count = count
        self.frames = frames
        self.size = size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = float(idx % 2 == 1)
        base = torch.rand(self.frames, 3, self.size, self.size) * 0.15
        if label == 1.0:
            for t in range(self.frames):
                x = int((t / max(self.frames - 1, 1)) * (self.size - 24))
                y = int(self.size / 2 - 12)
                base[t, :, y:y + 24, x:x + 24] += 0.75
        base = base.clamp(0.0, 1.0)
        return base, torch.tensor(label, dtype=torch.float32)
