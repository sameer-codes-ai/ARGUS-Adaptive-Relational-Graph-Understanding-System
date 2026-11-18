"""
ARGUS Live Interactive Surveillance Intelligence Platform & Web UI.

Features:
  - High-performance multi-threaded server (ThreadingHTTPServer)
  - Resilient native multipart/form-data video uploader (handles large MP4/AVI/MOV)
  - Pre-cached instant benchmark presets (Violence, Assault, Robbery, Normal Walkway)
  - Cinematic 16:9 surveillance player with HUD overlay, bounding boxes & P2P interaction graphs
  - Synchronized real-time Chart.js anomaly timeline with click-to-seek scrubbing
  - Forensic keyframe filmstrip with jump-to-frame navigation
  - Side-by-side / toggleable native HTML5 video player and AI Vision overlay
  - Full research paper drawer and one-click report generator
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
import webbrowser
from typing import Dict, List, Optional, Tuple
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import torch

from argus import ARGUS, AlertDispatcher, generate_dashboard_html
from argus.config import ARGUSConfig
from argus.data import load_clip_from_path, find_frame_files
from argus.checkpoint import load_checkpoint, model_state_dict, validate_component_metadata

CLIENT_DISCONNECT_ERRORS = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    ConnectionError,
)

# Global Model & State
GLOBAL_MODEL: Optional[ARGUS] = None
GLOBAL_CONFIG: Optional[ARGUSConfig] = None
MODEL_IS_TRAINED = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_LOCK = threading.RLock()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ANALYSIS_FRAMES = 256
UPLOAD_RETENTION_SECONDS = 60 * 60
ALLOWED_UPLOAD_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
PROJECT_ROOT = Path(__file__).resolve().parent
SAFE_OUTPUTS_EXTENSIONS = {
    '.html', '.htm', '.css', '.js', '.png', '.jpg', '.jpeg', '.svg',
    '.mp4', '.webm', '.avi', '.mov', '.csv',
}

# In-memory cache for presets to achieve sub-millisecond instant switching
PRESET_CACHE: Dict[str, dict] = {}

PRESET_DATASETS = [
    {
        "id": "violence_691",
        "title": "CAM-01 • Street Altercation & Physical Confrontation",
        "category": "Violent Incident",
        "dataset": "RealLifeViolence V_691",
        "expected": "CRITICAL INCIDENT (96.8%)",
        "badge_color": "#f43f5e",
        "badge_bg": "rgba(244, 63, 94, 0.12)",
        "description": "High-velocity physical altercation with multiple persons converging and anomalous kinetic strikes.",
        "path": r"demo_samples\sample_violence_V691.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\ViolenceP\V_691"
    },
    {
        "id": "violence_308",
        "title": "CAM-02 • Corridor Dispute & Close-Quarter Altercation",
        "category": "Violent Conflict",
        "dataset": "RealLifeViolence V_308",
        "expected": "CRITICAL INCIDENT (96.6%)",
        "badge_color": "#f43f5e",
        "badge_bg": "rgba(244, 63, 94, 0.12)",
        "description": "Escalating physical altercation between individuals in tight public corridor.",
        "path": r"demo_samples\sample_violence_V308.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\ViolenceP\V_308"
    },
    {
        "id": "robbery_087",
        "title": "CAM-03 • Commercial Facility Retail Incident",
        "category": "Property Incident",
        "dataset": "UCF-Crime Robbery087",
        "expected": "ELEVATED CONFLICT (68.8%)",
        "badge_color": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.12)",
        "description": "Rapid suspicious spatial convergence and forced item extraction in commercial environment.",
        "path": r"demo_samples\sample_robbery_087.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\organized\ucf\train\Robbery\Robbery087_x264"
    },
    {
        "id": "normal_344",
        "title": "CAM-04 • Pedestrian Thoroughfare & Sidewalk",
        "category": "Nominal Surveillance",
        "dataset": "RealLifeViolence NV_344",
        "expected": "NOMINAL SURVEILLANCE (0.7%)",
        "badge_color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.12)",
        "description": "Pedestrians walking normally along public sidewalk with safe physical distance.",
        "path": r"demo_samples\sample_normal_NV344.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\NonViolenceP\NV_344"
    },
    {
        "id": "normal_950",
        "title": "CAM-05 • Public Plaza & Open Ambient Space",
        "category": "Nominal Surveillance",
        "dataset": "RealLifeViolence NV_950",
        "expected": "NOMINAL SURVEILLANCE (0.5%)",
        "badge_color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.12)",
        "description": "Ambient public space with stationary and normal-paced pedestrians.",
        "path": r"demo_samples\sample_normal_NV950.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\NonViolenceP\NV_950"
    }
]


def load_model(checkpoint_path: Path, *, allow_random_model: bool = False, allow_unsafe_checkpoint: bool = False):
    global GLOBAL_MODEL, GLOBAL_CONFIG, MODEL_IS_TRAINED
    print(f"[*] Initializing ARGUS neural engine on {DEVICE.upper()} from {checkpoint_path}...")
    if not checkpoint_path.exists():
        if not allow_random_model:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Use --allow-random-model only for explicit development tests."
            )
        print(f"[!] DEVELOPMENT MODE: {checkpoint_path} not found; using untrained random weights.")
        GLOBAL_CONFIG = ARGUSConfig()
        GLOBAL_MODEL = ARGUS(config=GLOBAL_CONFIG).to(DEVICE)
        MODEL_IS_TRAINED = False
    else:
        state = load_checkpoint(checkpoint_path, map_location=DEVICE, allow_unsafe=allow_unsafe_checkpoint)
        saved_cfg = state.get('config', {})
        GLOBAL_CONFIG = ARGUSConfig.from_dict(saved_cfg, strict=True) if saved_cfg else ARGUSConfig().validate()
        GLOBAL_MODEL = ARGUS(config=GLOBAL_CONFIG).to(DEVICE)
        validate_component_metadata(state, GLOBAL_MODEL)
        GLOBAL_MODEL.load_state_dict(model_state_dict(state))
        epoch = state.get('epoch', '?') if isinstance(state, dict) else '?'
        print(f"[OK] Model weights loaded successfully (Trained Epoch {epoch})")
        MODEL_IS_TRAINED = True

    GLOBAL_MODEL.eval()


def parse_multipart_data(body: bytes, boundary: bytes) -> Dict:
    """Zero-copy-scan multipart/form-data parser for binary video files."""
    delim = b'--' + boundary
    delim_len = len(delim)
    form_data: Dict = {}
    pos = body.find(delim)
    while pos != -1:
        part_start = pos + delim_len
        if body[part_start:part_start + 2] == b'--':
            break
        if body[part_start:part_start + 2] == b'\r\n':
            part_start += 2
        next_pos = body.find(delim, part_start)
        part_end = next_pos if next_pos != -1 else len(body)
        header_end = body.find(b'\r\n\r\n', part_start, part_end)
        if header_end != -1:
            headers_text = body[part_start:header_end].decode('latin-1', errors='replace')
            payload_start = header_end + 4
            payload_end = part_end
            if payload_end >= payload_start + 2 and body[payload_end - 2:payload_end] == b'\r\n':
                payload_end -= 2
            name_match = re.search(r'name="([^"]+)"', headers_text)
            if name_match:
                field_name = name_match.group(1)
                filename_match = re.search(r'filename="([^"]+)"', headers_text)
                if filename_match:
                    form_data[field_name] = {
                        'filename': filename_match.group(1),
                        'content': body[payload_start:payload_end],
                    }
                else:
                    form_data[field_name] = body[payload_start:payload_end].decode('utf-8', errors='replace')
        pos = next_pos
    return form_data


def annotate_frame(
    frame_bgr: np.ndarray,
    boxes: Optional[torch.Tensor],
    score: float,
    threshold: float,
    frame_idx: int,
    total_frames: int,
    orig_idx: int,
    semantic_types: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    detector_impl: str = "yolo",
) -> np.ndarray:
    """
    Renders clean, high-definition surveillance OSD, actor bounding brackets, and relational scene graph edges.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    is_anomaly = score >= threshold
    if score >= 0.75:
        theme_bgr = (94, 63, 244)       # Crimson Red #f43f5e in BGR
        status_text = "CRITICAL ALERT: KINETIC ALTERCATION"
    elif score >= 0.50:
        theme_bgr = (11, 158, 245)      # Warm Amber #f59e0b in BGR
        status_text = "ELEVATED RISK: SPATIAL CONVERGENCE"
    elif score >= threshold:
        theme_bgr = (233, 165, 14)      # Sky Blue #0ea5e9 in BGR
        status_text = "MONITORED: UNUSUAL MOTION DETECTED"
    else:
        theme_bgr = (129, 185, 16)      # Emerald Green #10b981 in BGR
        status_text = "NOMINAL SURVEILLANCE ACTIVITY"

    # Extract & scale entity bounding boxes
    box_coords = []
    if boxes is not None and len(boxes) > 0:
        boxes_np = boxes.detach().cpu().numpy()
        for i in range(min(len(boxes_np), 8)):
            b = boxes_np[i]
            x1 = int(np.clip(b[0] * w, 0, w - 1))
            y1 = int(np.clip(b[1] * h, 0, h - 1))
            x2 = int(np.clip(b[2] * w, 0, w - 1))
            y2 = int(np.clip(b[3] * h, 0, h - 1))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_coords.append((x1, y1, x2, y2, cx, cy))

        # Draw Relational Scene Graph Edges (P2P, P2W, P2B interactions)
        for i in range(len(box_coords)):
            for j in range(i + 1, len(box_coords)):
                x1_i, y1_i, x2_i, y2_i, cx_i, cy_i = box_coords[i]
                x1_j, y1_j, x2_j, y2_j, cx_j, cy_j = box_coords[j]
                dist = np.hypot(cx_i - cx_j, cy_i - cy_j) / max(w, 1)

                if dist < 0.42:
                    ti = int(semantic_types[i].item() if hasattr(semantic_types[i], 'item') else semantic_types[i]) if semantic_types is not None and i < len(semantic_types) else 4
                    tj = int(semantic_types[j].item() if hasattr(semantic_types[j], 'item') else semantic_types[j]) if semantic_types is not None and j < len(semantic_types) else 4

                    is_p2w = (detector_impl == "yolo") and ((ti == 0 and tj == 3) or (ti == 3 and tj == 0))
                    is_p2b = (detector_impl == "yolo") and ((ti == 0 and tj == 2) or (ti == 2 and tj == 0))
                    is_p2p = (ti == 0 and tj == 0)

                    if is_p2w:
                        edge_color = (40, 40, 245)
                        thickness = 3
                        tag = "THREAT"
                    elif is_p2b:
                        edge_color = (11, 158, 245)
                        thickness = 2
                        tag = "P2B"
                    elif is_p2p:
                        edge_color = theme_bgr if is_anomaly else (120, 130, 145)
                        thickness = 2 if is_anomaly else 1
                        tag = "P2P" if is_anomaly else "PROX"
                    else:
                        edge_color = (120, 130, 145)
                        thickness = 1
                        tag = "REL"

                    cv2.line(annotated, (cx_i, cy_i), (cx_j, cy_j), edge_color, thickness, cv2.LINE_AA)
                    mx, my = (cx_i + cx_j) // 2, (cy_i + cy_j) // 2
                    cv2.putText(annotated, tag, (mx - 10, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, edge_color, 1, cv2.LINE_AA)

        # Draw Bounding Boxes with Semantic Labels & Confidences
        for idx, (x1, y1, x2, y2, cx, cy) in enumerate(box_coords):
            sem_id = 4
            conf_val = 1.0
            if semantic_types is not None and idx < len(semantic_types):
                sem_id = int(semantic_types[idx].item() if hasattr(semantic_types[idx], 'item') else semantic_types[idx])
            if scores is not None and idx < len(scores):
                conf_val = float(scores[idx].item() if hasattr(scores[idx], 'item') else scores[idx])

            if detector_impl in {"heuristic", "grid"}:
                # Fallback mode: honestly label broad heuristic proposals without fake weapon/bag claims
                if sem_id == 0:
                    label = f"PERSON [H]"
                elif sem_id == 1:
                    label = f"VEHICLE [H]"
                else:
                    label = f"OBJECT [H]"
                box_col = theme_bgr if is_anomaly else (180, 170, 140)
            else:
                # Real YOLO recognition
                if sem_id == 3:
                    label = f"KNIFE {conf_val:.2f}"
                    box_col = (40, 40, 245)  # Crimson threat
                elif sem_id == 2:
                    label = f"BAG {conf_val:.2f}"
                    box_col = (11, 158, 245)  # Amber
                elif sem_id == 0:
                    label = f"PERSON {conf_val:.2f}"
                    box_col = theme_bgr if is_anomaly else (180, 170, 140)
                elif sem_id == 1:
                    label = f"VEHICLE {conf_val:.2f}"
                    box_col = (200, 160, 50)
                else:
                    label = f"OBJECT {conf_val:.2f}"
                    box_col = (140, 150, 160)

            # Minimalist corner brackets for professional CCTV look
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, 1)
            tick = min(12, max(4, (x2 - x1) // 4))
            cv2.line(annotated, (x1, y1), (x1 + tick, y1), box_col, 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + tick), box_col, 2)
            cv2.line(annotated, (x2, y1), (x2 - tick, y1), box_col, 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + tick), box_col, 2)
            cv2.line(annotated, (x1, y2), (x1 + tick, y2), box_col, 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - tick), box_col, 2)
            cv2.line(annotated, (x2, y2), (x2 - tick, y2), box_col, 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - tick), box_col, 2)

            # Dark translucent label pill
            pill_w = max(72, len(label) * 8 + 10)
            if y1 >= 18:
                pill_y1, pill_y2 = y1 - 18, y1
                text_y = y1 - 4
            else:
                pill_y1, pill_y2 = y2, min(h, y2 + 18)
                text_y = min(h - 2, y2 + 14)
            cv2.rectangle(annotated, (x1, pill_y1), (min(w, x1 + pill_w), pill_y2), (10, 14, 20), -1)
            cv2.putText(annotated, label, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 230, 240), 1, cv2.LINE_AA)

    # ── Top Surveillance OSD Bar ──────────────────────────────────────────
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.line(annotated, (0, 42), (w, 42), (35, 45, 60), 1)

    # System Status & Frame Index
    cv2.putText(annotated, "CAM-01 [LIVE]  RTSP 1080P  H.264", (14, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 215), 1, cv2.LINE_AA)
    frame_meta = f"ARGUS EDGE-V2  |  SMPL {frame_idx + 1:02d}/{total_frames:02d}  |  KF #{orig_idx:03d}"
    cv2.putText(annotated, frame_meta, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 135, 155), 1, cv2.LINE_AA)

    # Threat Banner on Top-Right
    score_pct = f"ANOMALY: {score * 100:.1f}%"
    (pct_w, _), _ = cv2.getTextSize(score_pct, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    (st_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.putText(annotated, score_pct, (max(14, w - 14 - pct_w), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, status_text, (max(14, w - 14 - st_w), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, theme_bgr, 1, cv2.LINE_AA)

    # Bottom Progress Anomaly Line
    bar_w = int(w * min(score, 1.0))
    cv2.rectangle(annotated, (0, h - 3), (bar_w, h), theme_bgr, -1)

    return annotated


def compute_severity(peak_score: float, threshold: float = 0.35) -> str:
    """Classifies anomaly severity level with calibrated thresholds."""
    if peak_score >= 0.75:
        return "CRITICAL"
    elif peak_score >= 0.50:
        return "HIGH"
    elif peak_score >= threshold:
        return "ELEVATED"
    else:
        return "NORMAL"


def serve_file_ranged(handler: BaseHTTPRequestHandler, file_path: Path):
    """Handles HTTP 206 Partial Content range requests for smooth HTML5 video scrubbing."""
    if not file_path.exists() or not file_path.is_file():
        try:
            handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
        except CLIENT_DISCONNECT_ERRORS:
            pass
        return

    try:
        file_size = file_path.stat().st_size
        mime, _ = mimetypes.guess_type(str(file_path))
        content_type = mime or 'application/octet-stream'

        range_header = handler.headers.get('Range')
        if range_header and range_header.startswith('bytes='):
            try:
                range_val = range_header[6:].strip()
                parts = range_val.split('-')
                if not parts[0] and parts[1]:
                    suffix_length = int(parts[1])
                    if suffix_length <= 0:
                        raise ValueError("Invalid suffix byte range")
                    start = max(0, file_size - suffix_length)
                    end = file_size - 1
                else:
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if parts[1] else file_size - 1

                if start >= file_size or start > end:
                    handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    handler.send_header('Content-Range', f'bytes */{file_size}')
                    handler.end_headers()
                    return

                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                length = end - start + 1

                handler.send_response(HTTPStatus.PARTIAL_CONTENT)
                handler.send_header('Content-Type', content_type)
                handler.send_header('X-Content-Type-Options', 'nosniff')
                handler.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                handler.send_header('Content-Length', str(length))
                handler.send_header('Accept-Ranges', 'bytes')
                handler.end_headers()

                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    chunk_sz = 64 * 1024
                    while remaining > 0:
                        read_len = min(remaining, chunk_sz)
                        data = f.read(read_len)
                        if not data:
                            break
                        handler.wfile.write(data)
                        remaining -= len(data)
                return
            except CLIENT_DISCONNECT_ERRORS:
                # Normal client abort/disconnect during video scrub or navigation
                return
            except (ValueError, IndexError):
                handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid byte range")
                return

        # Full content fallback (200 OK)
        handler.send_response(HTTPStatus.OK)
        handler.send_header('Content-Type', content_type)
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('Content-Length', str(file_size))
        handler.send_header('Accept-Ranges', 'bytes')
        handler.end_headers()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                handler.wfile.write(chunk)
    except CLIENT_DISCONNECT_ERRORS:
        # Client closed socket during transmission
        pass
    except Exception:
        pass


def _decode_video_single_pass(
    source_path: Path,
    target_frames: int,
    image_size: int,
    disp_size: Tuple[int, int] = (640, 360),
) -> Optional[Tuple[torch.Tensor, List[np.ndarray], int, float, List[int]]]:
    """Decode model-resolution clip and display-resolution BGR frames in one VideoCapture pass."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        native_fps = float(fps_val) if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 25.0
        if total_native_frames <= 0:
            return None

        orig_indices = torch.linspace(0, total_native_frames - 1, target_frames).long().tolist()
        needed = set(orig_indices)
        max_needed = orig_indices[-1]

        tensor_map: Dict[int, torch.Tensor] = {}
        disp_map: Dict[int, np.ndarray] = {}
        frame_idx = 0
        while frame_idx <= max_needed:
            if not cap.grab():
                break
            if frame_idx in needed:
                ok_ret, frame = cap.retrieve()
                if ok_ret and frame is not None:
                    disp_map[frame_idx] = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA)
                    if frame.shape[0] != image_size or frame.shape[1] != image_size:
                        interp = cv2.INTER_AREA if frame.shape[0] > image_size else cv2.INTER_LINEAR
                        m_frame = cv2.resize(frame, (image_size, image_size), interpolation=interp)
                    else:
                        m_frame = frame
                    rgb = cv2.cvtColor(m_frame, cv2.COLOR_BGR2RGB)
                    tensor_map[frame_idx] = torch.from_numpy(rgb).permute(2, 0, 1)
            frame_idx += 1

        if len(tensor_map) != len(needed):
            return None

        clip = torch.stack([tensor_map[i] for i in orig_indices]).float() / 255.0
        disp_frames = [disp_map[i] for i in orig_indices]
        return clip, disp_frames, total_native_frames, native_fps, orig_indices
    finally:
        cap.release()


def process_video_inference(
    source_path: Path,
    threshold: float = 0.35,
    frames: int = 32,
    stream: bool = False
) -> Dict:
    """
    Executes ARGUS dual-layer pipeline and generates annotated base64 frames.
    """
    t_start = time.perf_counter()
    if GLOBAL_MODEL is None or GLOBAL_CONFIG is None:
        raise RuntimeError("Model has not been loaded")
    if not 1 <= frames <= MAX_ANALYSIS_FRAMES:
        raise ValueError(f"frames must be between 1 and {MAX_ANALYSIS_FRAMES}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    # Load video or frame directory in a single pass when possible
    t_dec0 = time.perf_counter()
    raw_frame_files = find_frame_files(source_path) if source_path.is_dir() else []
    target_size = GLOBAL_CONFIG.image_size if GLOBAL_CONFIG else 128
    disp_w, disp_h = 640, 360

    total_native_frames = 0
    native_fps = 25.0
    disp_frames: Optional[List[np.ndarray]] = None
    orig_frame_indices: Optional[List[int]] = None

    decoded = None
    if source_path.is_file() and source_path.suffix.lower() != '.pt':
        decoded = _decode_video_single_pass(
            source_path, target_frames=frames, image_size=target_size, disp_size=(disp_w, disp_h)
        )

    if decoded is not None:
        clip, disp_frames, total_native_frames, native_fps, orig_frame_indices = decoded
        duration_sec = total_native_frames / max(native_fps, 1.0)
    else:
        clip = load_clip_from_path(source_path, target_frames=frames, image_size=target_size)
        if raw_frame_files:
            orig_frame_indices = torch.linspace(0, len(raw_frame_files) - 1, frames).long().tolist()
            duration_sec = len(raw_frame_files) / max(native_fps, 1.0)
        else:
            orig_frame_indices = list(range(frames))
            duration_sec = frames / max(native_fps, 1.0)

    clip_batch = clip.unsqueeze(0).to(DEVICE)
    decode_time_ms = (time.perf_counter() - t_dec0) * 1000.0

    # Web accessible URL for native HTML5 video player (only if inside served static roots)
    video_url = None
    if source_path.is_file() and source_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        resolved_src = source_path.resolve()
        for static_root_name in ('demo_samples', 'outputs'):
            static_root = (PROJECT_ROOT / static_root_name).resolve()
            try:
                rel = resolved_src.relative_to(static_root)
                if static_root_name == 'outputs' and (
                    (rel.parts and rel.parts[0] == 'web_uploads')
                    or resolved_src.suffix.lower() not in SAFE_OUTPUTS_EXTENSIONS
                ):
                    break
                video_url = f"/{static_root_name}/{rel.as_posix()}"
                break
            except ValueError:
                continue

    # Neural inference
    t_inf0 = time.perf_counter()
    with MODEL_LOCK, torch.no_grad():
        out = GLOBAL_MODEL(clip_batch, stream=stream, stream_id=str(source_path.resolve()))
    infer_time_ms = (time.perf_counter() - t_inf0) * 1000.0

    valid_length = int(out.sequence_lengths[0].item())
    scores = out.scores[0, :valid_length].cpu().numpy().tolist()
    sampled_indices = out.sampled_indices[0].cpu().tolist() if isinstance(out.sampled_indices[0], torch.Tensor) else list(out.sampled_indices[0])

    peak_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    is_anomaly = bool(peak_score >= threshold)
    severity_str = compute_severity(peak_score, threshold=threshold)

    # High-definition display frames (640x360)
    annotated_frames = []
    timeline = []

    for idx, clip_pos in enumerate(sampled_indices):
        actual_video_frame = orig_frame_indices[clip_pos] if clip_pos < len(orig_frame_indices) else clip_pos
        timestamp_sec = actual_video_frame / max(native_fps, 1.0)
        sc = scores[idx]

        frame_bgr = None
        if disp_frames is not None and clip_pos < len(disp_frames):
            frame_bgr = disp_frames[clip_pos]
        elif raw_frame_files and actual_video_frame < len(raw_frame_files):
            try:
                frame_bgr = cv2.imread(str(raw_frame_files[actual_video_frame]))
            except Exception:
                pass

        if frame_bgr is None:
            t_frame = clip[clip_pos].permute(1, 2, 0).cpu().numpy()
            frame_bgr = (t_frame * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        if frame_bgr.shape[1] != disp_w or frame_bgr.shape[0] != disp_h:
            frame_bgr = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        boxes = out.boxes[0][idx] if (out.boxes and idx < len(out.boxes[0])) else None
        sem_types = out.semantic_types[0][idx] if (getattr(out, 'semantic_types', None) and idx < len(out.semantic_types[0])) else None
        det_scores = out.detection_scores[0][idx] if (getattr(out, 'detection_scores', None) and idx < len(out.detection_scores[0])) else None
        det_impl = getattr(out, 'detector_impl', 'yolo')

        annotated = annotate_frame(
            frame_bgr=frame_bgr,
            boxes=boxes,
            score=sc,
            threshold=threshold,
            frame_idx=idx,
            total_frames=len(sampled_indices),
            orig_idx=actual_video_frame,
            semantic_types=sem_types,
            scores=det_scores,
            detector_impl=det_impl,
        )

        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')

        mins, secs = divmod(int(timestamp_sec), 60)
        millis = int((timestamp_sec - int(timestamp_sec)) * 10)
        time_tag = f"{mins:02d}:{secs:02d}.{millis}"

        annotated_frames.append({
            "index": idx,
            "orig_idx": actual_video_frame,
            "time_sec": round(timestamp_sec, 2),
            "time_str": time_tag,
            "score": round(sc, 4),
            "is_anomaly": bool(sc >= threshold),
            "data_url": f"data:image/jpeg;base64,{b64_str}"
        })

        timeline.append({
            "time": round(timestamp_sec, 2),
            "frame": int(actual_video_frame),
            "score": round(float(sc), 4),
            "is_anomaly": bool(sc >= threshold)
        })

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    fps = float(len(sampled_indices) / max(infer_time_ms / 1000.0, 1e-4))
    frame_reduction = round((1.0 - (len(sampled_indices) / max(frames, 1))) * 100.0, 1)

    det_impl = getattr(out, 'detector_impl', 'yolo')
    dispatcher = AlertDispatcher(threshold=threshold)
    alert_event = dispatcher.dispatch(
        video_source=str(source_path),
        timeline_scores=[round(s, 4) for s in scores],
        keyframe_indices=sampled_indices,
        boxes_per_frame=out.boxes[0] if out.boxes else None,
        semantic_types_per_frame=out.semantic_types[0] if getattr(out, 'semantic_types', None) else None,
        detection_scores_per_frame=out.detection_scores[0] if getattr(out, 'detection_scores', None) else None,
        edge_indices_per_frame=out.edge_indices[0] if getattr(out, 'edge_indices', None) else None,
        edge_attrs_per_frame=out.edge_attrs[0] if getattr(out, 'edge_attrs', None) else None,
        detector_impl=det_impl,
        class_ids_per_frame=out.class_ids[0] if getattr(out, 'class_ids', None) else None,
    )

    return {
        "status": "success",
        "video_name": source_path.name,
        "video_url": video_url,
        "duration_sec": round(duration_sec, 2),
        "native_fps": round(native_fps, 1),
        "total_native_frames": total_native_frames or len(raw_frame_files) or frames,
        "is_anomaly": is_anomaly,
        "severity": severity_str,
        "peak_score": round(peak_score, 4),
        "mean_score": round(mean_score, 4),
        "threshold": threshold,
        "timeline": timeline,
        "timeline_scores": [round(s, 4) for s in scores],
        "sampled_indices": sampled_indices,
        "total_sampled": len(sampled_indices),
        "total_input_frames": frames,
        "frame_reduction_pct": frame_reduction,
        "fps": round(fps, 1),
        "timings": {
            "decode_ms": round(decode_time_ms, 1),
            "infer_ms": round(infer_time_ms, 1),
            "total_ms": round(total_time_ms, 1)
        },
        "detector_implementation": det_impl,
        "detector_name": "YOLOv11 Semantic" if det_impl == "yolo" else "Heuristic Proposals",
        "person_count": alert_event.person_count if alert_event else 0,
        "vehicle_count": alert_event.vehicle_count if alert_event else 0,
        "bag_count": alert_event.bag_count if alert_event else 0,
        "weapon_count": alert_event.weapon_count if alert_event else 0,
        "other_count": alert_event.other_count if alert_event else 0,
        "object_counts": alert_event.object_counts if (alert_event and alert_event.object_counts) else {},
        "person_weapon_edges": alert_event.person_weapon_edges if alert_event else 0,
        "num_person_interactions": alert_event.num_person_interactions if alert_event else 0,
        "evidence_summary": alert_event.summary if alert_event else "",
        "evidence_details": alert_event.evidence_details if (alert_event and alert_event.evidence_details) else [],
        "num_entities": max((len(boxes) for boxes in out.boxes[0]), default=0) if out.boxes else 0,
        "frames": annotated_frames
    }


def warm_preset_cache():
    """Pre-runs inference on the curated presets for zero-latency instant demo."""
    print("[*] Warming benchmark preset cache for instant evaluation...")
    for p in PRESET_DATASETS:
        target_path = Path(p['path'])
        if not target_path.exists():
            target_path = Path(p.get('fallback_path', ''))

        if target_path.exists():
            try:
                res = process_video_inference(target_path, threshold=0.35, frames=24)
                PRESET_CACHE[p['id']] = res
                print(f"  [OK] Pre-cached {p['id']} ({target_path.name}) -> Peak Score: {res['peak_score']}")
            except Exception as e:
                print(f"  [!] Failed to pre-cache {p['id']}: {e}")
    print("[OK] Benchmark cache ready.\n")


# ── Frontend HTML/CSS/JS Single Page Application ─────────────────────────
def get_frontend_html() -> str:
    """Load frontend.html and inject preset data."""
    html_path = Path(__file__).parent / 'frontend.html'
    raw = html_path.read_text(encoding='utf-8')
    return raw.replace('%%PRESETS_JSON%%', json.dumps(PRESET_DATASETS))


def resolve_static_path(request_path: str) -> Optional[Path]:
    decoded = urllib.parse.unquote(request_path)
    segments = [segment for segment in decoded.split('/') if segment]
    if len(segments) < 2 or segments[0] not in {'demo_samples', 'outputs', 'docs'}:
        return None
    if segments[0] == 'outputs' and segments[1] == 'web_uploads':
        return None
    root = (PROJECT_ROOT / segments[0]).resolve()
    candidate = (root / Path(*segments[1:])).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    if segments[0] == 'outputs' and candidate.suffix.lower() not in SAFE_OUTPUTS_EXTENSIONS:
        return None
    return candidate


def cleanup_expired_uploads(upload_dir: Path) -> None:
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for entry in upload_dir.iterdir() if upload_dir.exists() else []:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


class SafeThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that gracefully handles client disconnects (video scrubs, tab closes)."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type is not None:
            if issubclass(exc_type, CLIENT_DISCONNECT_ERRORS):
                return
            if isinstance(exc_val, OSError) and getattr(exc_val, 'winerror', None) in (10053, 10054):
                return
        super().handle_error(request, client_address)


class ARGUSWebHandler(BaseHTTPRequestHandler):
    """Robust multi-threaded HTTP request handler for the ARGUS web demo."""

    def log_message(self, format, *args):
        # Clean logging
        pass

    def send_json_response(self, data: dict, status: int = HTTPStatus.OK):
        try:
            body = json.dumps(data).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except CLIENT_DISCONNECT_ERRORS:
            pass

    def do_OPTIONS(self):
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header('Allow', 'GET, POST, OPTIONS')
            self.end_headers()
        except CLIENT_DISCONNECT_ERRORS:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ('/', '/index.html'):
            try:
                body = get_frontend_html().encode('utf-8')
            except Exception as e:
                body = f"<h1>ARGUS Frontend Loading Error</h1><pre>{html.escape(str(e))}</pre>".encode('utf-8')
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except CLIENT_DISCONNECT_ERRORS:
                pass
            return

        elif path == '/api/presets':
            self.send_json_response({"status": "success", "presets": PRESET_DATASETS})
            return

        elif path == '/api/paper_summary':
            summary = {
                "title": "Adaptive Relational Graph Understanding for Surveillance (ARGUS)",
                "authors": "Omkar Joshi, Sameer Kumar",
                "institution": "School of Computer Science and Engineering, Vellore Institute of Technology, Vellore",
                "benchmark": {
                    "roc_auc": 0.7949,
                    "average_precision": 0.7449,
                    "accuracy": 0.7273,
                    "f1_score": 0.7038,
                    "speed_cpu_fps": 13.6
                }
            }
            self.send_json_response(summary)
            return

        # Serve media files, static docs, or dashboards with HTTP 206 Partial Content range support
        elif path.startswith('/demo_samples/') or path.startswith('/outputs/') or path.startswith('/docs/'):
            local_file = resolve_static_path(path)
            if local_file is not None:
                serve_file_ranged(self, local_file)
                return

        try:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except CLIENT_DISCONNECT_ERRORS:
            pass

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        origin = self.headers.get('Origin', '').strip()
        if origin:
            parsed_origin = urllib.parse.urlparse(origin)
            host_header = self.headers.get('Host', '').strip()
            if not parsed_origin.netloc or (host_header and parsed_origin.netloc.lower() != host_header.lower()):
                self.send_json_response({"status": "error", "message": "Origin mismatch"}, status=HTTPStatus.FORBIDDEN)
                return

        if path == '/api/analyze':
            content_type = self.headers.get('Content-Type', '')

            # Case A: Preset JSON request
            if 'application/json' in content_type:
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                except (ValueError, TypeError):
                    self.send_json_response({"status": "error", "message": "Invalid Content-Length header"}, status=400)
                    return
                if content_length <= 0 or content_length > 1024 * 1024:
                    self.send_json_response({"status": "error", "message": "Invalid JSON request size"}, status=413)
                    return
                raw_body = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw_body.decode('utf-8'))
                except Exception:
                    self.send_json_response({"status": "error", "message": "Malformed JSON payload"}, status=400)
                    return

                preset_id = payload.get('preset_id', 'violence_691')
                try:
                    threshold = float(payload.get('threshold', 0.35))
                    frames = int(payload.get('frames', 24))
                except (ValueError, TypeError):
                    self.send_json_response({"status": "error", "message": "Invalid frames or threshold"}, status=400)
                    return
                if not 1 <= frames <= MAX_ANALYSIS_FRAMES or not 0.0 <= threshold <= 1.0:
                    self.send_json_response({"status": "error", "message": "Invalid frames or threshold"}, status=400)
                    return
                stream = bool(payload.get('stream', False))

                # Return pre-cached result if available and using default parameters
                if preset_id in PRESET_CACHE and abs(threshold - 0.35) < 1e-4 and frames == 24 and not stream:
                    self.send_json_response(PRESET_CACHE[preset_id])
                    return

                # Find preset path
                matched = next((p for p in PRESET_DATASETS if p['id'] == preset_id), None)
                if not matched:
                    self.send_json_response({"status": "error", "message": f"Unknown preset: {preset_id}"}, status=404)
                    return

                target_path = Path(matched['path'])
                if not target_path.exists():
                    target_path = Path(matched.get('fallback_path', ''))

                if not target_path.exists():
                    self.send_json_response({"status": "error", "message": f"Media not found: {matched['path']}"}, status=404)
                    return

                try:
                    res = process_video_inference(target_path, threshold=threshold, frames=frames, stream=stream)
                    self.send_json_response(res)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    self.send_json_response({"status": "error", "message": "Internal inference error"}, status=500)
                return

            # Case B: Multipart file upload (Handles binary MP4/AVI/MOV up to 100MB)
            elif 'multipart/form-data' in content_type:
                upload_dir = PROJECT_ROOT / 'outputs' / 'web_uploads'
                upload_dir.mkdir(parents=True, exist_ok=True)
                cleanup_expired_uploads(upload_dir)

                # Extract boundary
                boundary_match = re.search(r'boundary=([^;]+)', content_type)
                if not boundary_match:
                    self.send_json_response({"status": "error", "message": "No boundary in multipart header"}, status=400)
                    return
                boundary = boundary_match.group(1).strip().strip('"').encode('utf-8')

                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                except (ValueError, TypeError):
                    self.send_json_response({"status": "error", "message": "Invalid Content-Length header"}, status=400)
                    return
                if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
                    self.send_json_response({"status": "error", "message": "Upload must be between 1 byte and 100 MB"}, status=413)
                    return
                raw_body = self.rfile.read(content_length)

                parsed_form = parse_multipart_data(raw_body, boundary)
                video_item = parsed_form.get('video_file')
                if not video_item or not isinstance(video_item, dict):
                    self.send_json_response({"status": "error", "message": "No video file found in upload."}, status=400)
                    return

                filename = video_item.get('filename') or "uploaded_surveillance.mp4"
                file_bytes = video_item.get('content', b'')
                try:
                    threshold = float(parsed_form.get('threshold', 0.35))
                    # For uploaded videos, default to 32 dense frames for high temporal resolution
                    frames = int(parsed_form.get('frames', 32))
                except (ValueError, TypeError):
                    self.send_json_response({"status": "error", "message": "Invalid frames or threshold"}, status=400)
                    return
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
                    self.send_json_response({"status": "error", "message": "Unsupported video file type"}, status=415)
                    return
                if not file_bytes:
                    self.send_json_response({"status": "error", "message": "Uploaded file is empty"}, status=400)
                    return
                if not 1 <= frames <= MAX_ANALYSIS_FRAMES or not 0.0 <= threshold <= 1.0:
                    self.send_json_response({"status": "error", "message": "Invalid frames or threshold"}, status=400)
                    return

                saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
                with open(saved_path, 'wb') as f:
                    f.write(file_bytes)

                print(f"[*] Received video upload: {filename} ({len(file_bytes)} bytes) -> Saved to {saved_path}")

                try:
                    res = process_video_inference(saved_path, threshold=threshold, frames=frames, stream=False)
                    self.send_json_response(res)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    self.send_json_response({"status": "error", "message": "Inference failed"}, status=500)
                return

        try:
            self.send_error(HTTPStatus.NOT_FOUND, "API Endpoint Not Found")
        except CLIENT_DISCONNECT_ERRORS:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run ARGUS Live Web Demo.")
    parser.add_argument('--port', type=int, default=7860, help='Web server port (default: 7860).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server bind host (default: localhost only).')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/checkpoints_rlvd_full/best.pt'), help='Model checkpoint.')
    parser.add_argument('--allow-random-model', action='store_true', help='Explicitly allow untrained random weights for development.')
    parser.add_argument('--allow-unsafe-checkpoint', action='store_true', help='Allow a trusted legacy pickle checkpoint.')
    parser.add_argument('--open-browser', action='store_true', help='Open the web UI in the default browser once listening.')
    args = parser.parse_args()

    # Preload PyTorch model
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    load_model(
        args.checkpoint,
        allow_random_model=args.allow_random_model,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
    )

    # Pre-cache presets for instant 0ms latency switching
    warm_preset_cache()

    server_address = (args.host, args.port)
    httpd = SafeThreadingHTTPServer(server_address, ARGUSWebHandler)
    if args.open_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}")

    print("\n" + "=" * 66)
    print("      [+]  ARGUS AUTONOMOUS SURVEILLANCE PLATFORM ONLINE  [+]      ")
    print("=" * 66)
    print(f"  Local Access:      http://127.0.0.1:{args.port}")
    print(f"  Network Access:    http://localhost:{args.port}")
    active_model = "Loaded trained checkpoint" if MODEL_IS_TRAINED else "UNTRAINED DEVELOPMENT MODEL"
    print(f"  Active Model:      {active_model}")
    print(f"  Device Compute:    {DEVICE.upper()}")
    print("=" * 66)
    print("  Server is non-blocking & multi-threaded. Press Ctrl+C to terminate.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server.")
        httpd.server_close()


if __name__ == '__main__':
    main()
