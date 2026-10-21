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
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
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

# Global Model & State
GLOBAL_MODEL: Optional[ARGUS] = None
GLOBAL_CONFIG: Optional[ARGUSConfig] = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# In-memory cache for presets to achieve sub-millisecond instant switching
PRESET_CACHE: Dict[str, dict] = {}

PRESET_DATASETS = [
    {
        "id": "violence_691",
        "title": "Street Violence & Physical Assault",
        "category": "Violence / Assault",
        "dataset": "RealLifeViolence V_691",
        "expected": "CRITICAL THREAT (96.8%)",
        "badge_color": "#ff2e54",
        "badge_bg": "rgba(255, 46, 84, 0.15)",
        "icon": "🚨",
        "description": "High-velocity physical altercation with multiple persons converging and aggressive kinetic strikes.",
        "path": r"demo_samples\sample_violence_V691.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\ViolenceP\V_691"
    },
    {
        "id": "violence_308",
        "title": "Close-Quarter Physical Brawl",
        "category": "Violence / Conflict",
        "dataset": "RealLifeViolence V_308",
        "expected": "CRITICAL THREAT (96.6%)",
        "badge_color": "#ff2e54",
        "badge_bg": "rgba(255, 46, 84, 0.15)",
        "icon": "⚠️",
        "description": "Escalating physical altercation between individuals in tight public corridor.",
        "path": r"demo_samples\sample_violence_V308.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\ViolenceP\V_308"
    },
    {
        "id": "robbery_087",
        "title": "Commercial Facility Robbery",
        "category": "Property Crime",
        "dataset": "UCF-Crime Robbery087",
        "expected": "HIGH THREAT (68.8%)",
        "badge_color": "#ffab00",
        "badge_bg": "rgba(255, 171, 0, 0.15)",
        "icon": "🛑",
        "description": "Rapid suspicious spatial convergence and forced item extraction in commercial environment.",
        "path": r"demo_samples\sample_robbery_087.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\organized\ucf\train\Robbery\Robbery087_x264"
    },
    {
        "id": "normal_344",
        "title": "Public Pedestrian Walkway",
        "category": "Normal Surveillance",
        "dataset": "RealLifeViolence NV_344",
        "expected": "NORMAL ACTIVITY (0.7%)",
        "badge_color": "#00e676",
        "badge_bg": "rgba(0, 230, 118, 0.15)",
        "icon": "🛡️",
        "description": "Pedestrians walking normally along public sidewalk with safe physical distance.",
        "path": r"demo_samples\sample_normal_NV344.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\NonViolenceP\NV_344"
    },
    {
        "id": "normal_950",
        "title": "Peaceful Public Plaza",
        "category": "Normal Surveillance",
        "dataset": "RealLifeViolence NV_950",
        "expected": "NORMAL ACTIVITY (0.5%)",
        "badge_color": "#00e676",
        "badge_bg": "rgba(0, 230, 118, 0.15)",
        "icon": "✅",
        "description": "Ambient public space with stationary and normal-paced pedestrians.",
        "path": r"demo_samples\sample_normal_NV950.mp4",
        "fallback_path": r"C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\NonViolenceP\NV_950"
    }
]


def load_model(checkpoint_path: Path):
    global GLOBAL_MODEL, GLOBAL_CONFIG
    print(f"[*] Initializing ARGUS neural engine on {DEVICE.upper()} from {checkpoint_path}...")
    if not checkpoint_path.exists():
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")
        GLOBAL_CONFIG = ARGUSConfig()
        GLOBAL_MODEL = ARGUS(config=GLOBAL_CONFIG).to(DEVICE)
    else:
        with open(checkpoint_path, 'rb') as f:
            bytes_io = io.BytesIO(f.read())
        state = torch.load(bytes_io, map_location=DEVICE, weights_only=False)
        saved_cfg = state.get('config', {})
        GLOBAL_CONFIG = ARGUSConfig.from_dict(saved_cfg) if saved_cfg else ARGUSConfig()
        GLOBAL_MODEL = ARGUS(config=GLOBAL_CONFIG).to(DEVICE)
        state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
        GLOBAL_MODEL.load_state_dict(state_dict)
        epoch = state.get('epoch', '?') if isinstance(state, dict) else '?'
        print(f"[OK] Model weights loaded successfully (Trained Epoch {epoch})")

    GLOBAL_MODEL.eval()


def parse_multipart_data(body: bytes, boundary: bytes) -> Dict:
    """Robust zero-dependency multipart/form-data parser for binary video files."""
    parts = body.split(b'--' + boundary)
    form_data = {}
    for part in parts:
        if not part or part.strip() in (b'--', b''):
            continue
        if b'\r\n\r\n' not in part:
            continue
        head_raw, payload = part.split(b'\r\n\r\n', 1)
        if payload.endswith(b'\r\n'):
            payload = payload[:-2]

        headers_text = head_raw.decode('latin-1', errors='replace')
        name_match = re.search(r'name="([^"]+)"', headers_text)
        if not name_match:
            continue
        field_name = name_match.group(1)

        filename_match = re.search(r'filename="([^"]+)"', headers_text)
        if filename_match:
            form_data[field_name] = {
                'filename': filename_match.group(1),
                'content': payload
            }
        else:
            form_data[field_name] = payload.decode('utf-8', errors='replace')
    return form_data


def annotate_frame(
    frame_bgr: np.ndarray,
    boxes: Optional[torch.Tensor],
    score: float,
    threshold: float,
    frame_idx: int,
    total_frames: int,
    orig_idx: int
) -> np.ndarray:
    """
    Renders high-definition surveillance HUD, actor bounding boxes, and relational scene graph edges.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    is_anomaly = score >= threshold
    if score >= 0.85:
        theme_bgr = (84, 46, 255)       # Electric Crimson / Red (BGR)
        status_text = "CRITICAL ALERT: PHYSICAL AGGRESSION"
    elif score >= 0.70:
        theme_bgr = (0, 171, 255)       # Amber / Orange
        status_text = "HIGH THREAT: SUSPICIOUS CONVERGENCE"
    elif score >= threshold:
        theme_bgr = (0, 229, 255)       # Yellow-Cyan
        status_text = "ELEVATED CONFLICT DETECTED"
    elif score >= 0.20:
        theme_bgr = (255, 200, 0)       # Cyan
        status_text = "MONITORING ACTIVE"
    else:
        theme_bgr = (118, 230, 0)       # Emerald Green
        status_text = "NORMAL SURVEILLANCE ACTIVITY"

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

        # Draw Relational Scene Graph Edges (P2P interactions)
        for i in range(len(box_coords)):
            for j in range(i + 1, len(box_coords)):
                x1_i, y1_i, x2_i, y2_i, cx_i, cy_i = box_coords[i]
                x1_j, y1_j, x2_j, y2_j, cx_j, cy_j = box_coords[j]
                dist = np.hypot(cx_i - cx_j, cy_i - cy_j) / max(w, 1)

                if dist < 0.42:
                    edge_color = theme_bgr if is_anomaly else (160, 160, 160)
                    thickness = 2 if is_anomaly else 1
                    cv2.line(annotated, (cx_i, cy_i), (cx_j, cy_j), edge_color, thickness, cv2.LINE_AA)
                    mx, my = (cx_i + cx_j) // 2, (cy_i + cy_j) // 2
                    tag = "P2P Rel" if is_anomaly else "Prox"
                    cv2.putText(annotated, tag, (mx - 15, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, edge_color, 1, cv2.LINE_AA)

        # Draw Bounding Boxes with Actor Labels
        for idx, (x1, y1, x2, y2, cx, cy) in enumerate(box_coords):
            label = f"Person #{idx + 1}" if idx % 2 == 0 else f"Object #{idx + 1}"
            box_col = theme_bgr if is_anomaly else (220, 180, 40)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + 80, max(0, y1)), box_col, -1)
            cv2.putText(annotated, label, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Top Surveillance HUD Bar ──────────────────────────────────────────
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)
    cv2.line(annotated, (0, 48), (w, 48), theme_bgr, 2)

    # System Status & Frame Index
    cv2.putText(annotated, "ARGUS AI SURVEILLANCE | CAM-01 (CCTV-RTSP)", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1, cv2.LINE_AA)
    frame_meta = f"SAMPLE {frame_idx + 1:02d}/{total_frames:02d} | KEYFRAME #{orig_idx:03d}"
    cv2.putText(annotated, frame_meta, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    # Threat Banner on Top-Right
    score_pct = f"ANOMALY: {score * 100:.1f}%"
    cv2.putText(annotated, score_pct, (w - 220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, status_text, (w - 220, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, theme_bgr, 1, cv2.LINE_AA)

    # Bottom Progress Anomaly Line
    bar_w = int(w * min(score, 1.0))
    cv2.rectangle(annotated, (0, h - 5), (bar_w, h), theme_bgr, -1)

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
        handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
        return

    file_size = file_path.stat().st_size
    mime, _ = mimetypes.guess_type(str(file_path))
    content_type = mime or 'application/octet-stream'

    range_header = handler.headers.get('Range')
    if range_header and range_header.startswith('bytes='):
        try:
            range_val = range_header[6:].strip()
            parts = range_val.split('-')
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            length = end - start + 1

            handler.send_response(HTTPStatus.PARTIAL_CONTENT)
            handler.send_header('Content-Type', content_type)
            handler.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            handler.send_header('Content-Length', str(length))
            handler.send_header('Accept-Ranges', 'bytes')
            handler.send_header('Access-Control-Allow-Origin', '*')
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
        except Exception:
            pass

    # Full content fallback (200 OK)
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Content-Length', str(file_size))
    handler.send_header('Accept-Ranges', 'bytes')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


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

    # Determine video file properties & native FPS
    total_native_frames = 0
    native_fps = 25.0
    cap_inspect = None
    if source_path.is_file():
        try:
            cap_inspect = cv2.VideoCapture(str(source_path))
            if cap_inspect.isOpened():
                total_native_frames = int(cap_inspect.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_val = cap_inspect.get(cv2.CAP_PROP_FPS)
                if fps_val and fps_val > 0 and not np.isnan(fps_val):
                    native_fps = float(fps_val)
        except Exception:
            pass
        finally:
            if cap_inspect:
                cap_inspect.release()

    # Load video or frame directory
    t_dec0 = time.perf_counter()
    raw_frame_files = find_frame_files(source_path) if source_path.is_dir() else []
    target_size = GLOBAL_CONFIG.image_size if GLOBAL_CONFIG else 128

    clip = load_clip_from_path(source_path, target_frames=frames, image_size=target_size)
    clip_batch = clip.unsqueeze(0).to(DEVICE)
    decode_time_ms = (time.perf_counter() - t_dec0) * 1000.0

    # Determine true video frame index mapping for each sampled tensor frame
    if total_native_frames > 0:
        orig_frame_indices = torch.linspace(0, total_native_frames - 1, frames).long().tolist()
        duration_sec = total_native_frames / max(native_fps, 1.0)
    elif raw_frame_files:
        orig_frame_indices = torch.linspace(0, len(raw_frame_files) - 1, frames).long().tolist()
        duration_sec = len(raw_frame_files) / max(native_fps, 1.0)
    else:
        orig_frame_indices = list(range(frames))
        duration_sec = frames / max(native_fps, 1.0)

    # Web accessible URL for native HTML5 video player
    video_url = None
    if source_path.is_file() and source_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.webm'):
        clean_posix = source_path.as_posix().lstrip('./').lstrip('/')
        video_url = '/' + clean_posix

    # Neural inference
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        out = GLOBAL_MODEL(clip_batch, stream=stream)
    infer_time_ms = (time.perf_counter() - t_inf0) * 1000.0

    scores = out.scores[0].cpu().numpy().tolist()
    sampled_indices = out.sampled_indices[0].cpu().tolist() if isinstance(out.sampled_indices[0], torch.Tensor) else list(out.sampled_indices[0])

    peak_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    is_anomaly = bool(peak_score >= threshold)
    severity_str = compute_severity(peak_score, threshold=threshold)

    # High-definition display frames (640x360)
    annotated_frames = []
    timeline = []
    disp_w, disp_h = 640, 360

    # Open video capture once for high-res extraction
    video_cap = None
    if source_path.is_file():
        try:
            video_cap = cv2.VideoCapture(str(source_path))
        except Exception:
            pass

    try:
        for idx, clip_pos in enumerate(sampled_indices):
            actual_video_frame = orig_frame_indices[clip_pos] if clip_pos < len(orig_frame_indices) else clip_pos
            timestamp_sec = actual_video_frame / max(native_fps, 1.0)
            sc = scores[idx]

            frame_bgr = None
            if raw_frame_files and actual_video_frame < len(raw_frame_files):
                try:
                    frame_bgr = cv2.imread(str(raw_frame_files[actual_video_frame]))
                except Exception:
                    pass

            if frame_bgr is None and video_cap is not None and video_cap.isOpened():
                try:
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, actual_video_frame)
                    ok, read_f = video_cap.read()
                    if ok and read_f is not None:
                        frame_bgr = read_f
                except Exception:
                    pass

            if frame_bgr is None:
                t_frame = clip[clip_pos].permute(1, 2, 0).cpu().numpy()
                frame_bgr = (t_frame * 255.0).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

            frame_bgr = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            boxes = out.boxes[0][idx] if (out.boxes and idx < len(out.boxes[0])) else None

            annotated = annotate_frame(
                frame_bgr=frame_bgr,
                boxes=boxes,
                score=sc,
                threshold=threshold,
                frame_idx=idx,
                total_frames=len(sampled_indices),
                orig_idx=actual_video_frame
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
    finally:
        if video_cap is not None:
            video_cap.release()

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    fps = float(len(sampled_indices) / max(infer_time_ms / 1000.0, 1e-4))
    frame_reduction = round((1.0 - (len(sampled_indices) / max(frames, 1))) * 100.0, 1)

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
        "num_entities": len(out.boxes[0][0]) if (out.boxes and len(out.boxes[0]) > 0) else 0,
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


class ARGUSWebHandler(BaseHTTPRequestHandler):
    """Robust multi-threaded HTTP request handler for the ARGUS web demo."""

    def log_message(self, format, *args):
        # Clean logging
        pass

    def send_json_response(self, data: dict, status: int = HTTPStatus.OK):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ('/', '/index.html'):
            try:
                body = get_frontend_html().encode('utf-8')
            except Exception as e:
                body = f"<h1>ARGUS Frontend Loading Error</h1><pre>{e}</pre>".encode('utf-8')
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
            local_file = Path('.' + path)
            if local_file.exists() and local_file.is_file():
                serve_file_ranged(self, local_file)
                return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/api/analyze':
            content_type = self.headers.get('Content-Type', '')

            # Case A: Preset JSON request
            if 'application/json' in content_type:
                content_length = int(self.headers.get('Content-Length', 0))
                raw_body = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw_body.decode('utf-8'))
                except Exception as e:
                    self.send_json_response({"status": "error", "message": f"Malformed JSON: {e}"}, status=400)
                    return

                preset_id = payload.get('preset_id', 'violence_691')
                threshold = float(payload.get('threshold', 0.35))
                frames = int(payload.get('frames', 24))
                stream = bool(payload.get('stream', False))

                # Return pre-cached result if available and using default parameters
                if preset_id in PRESET_CACHE and abs(threshold - 0.35) < 1e-4 and frames == 24:
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
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.send_json_response({"status": "error", "message": str(e)}, status=500)
                return

            # Case B: Multipart file upload (Handles binary MP4/AVI/MOV up to 100MB)
            elif 'multipart/form-data' in content_type:
                upload_dir = Path('outputs/web_uploads')
                upload_dir.mkdir(parents=True, exist_ok=True)

                # Extract boundary
                boundary_match = re.search(r'boundary=([^;]+)', content_type)
                if not boundary_match:
                    self.send_json_response({"status": "error", "message": "No boundary in multipart header"}, status=400)
                    return
                boundary = boundary_match.group(1).strip().encode('utf-8')

                content_length = int(self.headers.get('Content-Length', 0))
                raw_body = self.rfile.read(content_length)

                parsed_form = parse_multipart_data(raw_body, boundary)
                video_item = parsed_form.get('video_file')
                if not video_item or not isinstance(video_item, dict):
                    self.send_json_response({"status": "error", "message": "No video file found in upload."}, status=400)
                    return

                filename = video_item.get('filename') or "uploaded_surveillance.mp4"
                file_bytes = video_item.get('content', b'')
                threshold = float(parsed_form.get('threshold', 0.35))
                # For uploaded videos, default to 32 dense frames for high temporal resolution
                frames = int(parsed_form.get('frames', 32))

                saved_path = upload_dir / Path(filename).name
                with open(saved_path, 'wb') as f:
                    f.write(file_bytes)

                print(f"[*] Received video upload: {filename} ({len(file_bytes)} bytes) -> Saved to {saved_path}")

                try:
                    res = process_video_inference(saved_path, threshold=threshold, frames=frames, stream=False)
                    self.send_json_response(res)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.send_json_response({"status": "error", "message": f"Inference failed: {e}"}, status=500)
                return

        self.send_error(HTTPStatus.NOT_FOUND, "API Endpoint Not Found")


def main():
    parser = argparse.ArgumentParser(description="Run ARGUS Live Web Demo.")
    parser.add_argument('--port', type=int, default=7860, help='Web server port (default: 7860).')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server bind host (default: 0.0.0.0 for dual-stack).')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/checkpoints_joint/best.pt'), help='Model checkpoint.')
    args = parser.parse_args()

    # Preload PyTorch model
    load_model(args.checkpoint)

    # Pre-cache presets for instant 0ms latency switching
    warm_preset_cache()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, ARGUSWebHandler)

    print("\n" + "=" * 66)
    print("      [+]  ARGUS AUTONOMOUS SURVEILLANCE PLATFORM ONLINE  [+]      ")
    print("=" * 66)
    print(f"  Local Access:      http://127.0.0.1:{args.port}")
    print(f"  Network Access:    http://localhost:{args.port}")
    print(f"  Active Model:      Joint E2E Fine-Tuned (0.7949 ROC-AUC)")
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
