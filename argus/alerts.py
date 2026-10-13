"""
Real-Time Alert Dispatcher & Dashboard Interface for ARGUS.

Implements the final stages of the ARGUS pipeline (as required by the system feedback):
  [ Real-Time Anomaly Scoring & Explainability Engine ]
                       │
                       ▼
  [ Real-Time Alert Dispatcher & Dashboard Interface ]

Features:
  - AlertDispatcher: Rate-limited, severity-ranked real-time alert engine.
  - DashboardExporter: Standalone, interactive HTML/CSS/JS dashboard generator
    for visualizing timelines, scene graph interactions, and anomaly keyframes.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional


class AlertSeverity(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AlertEvent:
    timestamp: float
    video_source: str
    peak_score: float
    mean_score: float
    severity: AlertSeverity
    keyframe_indices: List[int]
    timeline_scores: List[float]
    num_detected_objects: int
    num_person_interactions: int
    is_anomaly: bool
    summary: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


class AlertDispatcher:
    """
    Real-Time Alert Dispatcher with configurable thresholds, cooldown windows,
    and extensible notification callbacks.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        cooldown_seconds: float = 5.0,
        log_file: Optional[Path] = None,
        callbacks: Optional[List[Callable[[AlertEvent], None]]] = None
    ):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.log_file = Path(log_file) if log_file else None
        self.callbacks = callbacks or []
        self._last_alert_time: float = 0.0

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def compute_severity(self, peak_score: float) -> AlertSeverity:
        if peak_score < self.threshold:
            return AlertSeverity.NORMAL
        elif peak_score < 0.65:
            return AlertSeverity.LOW
        elif peak_score < 0.80:
            return AlertSeverity.MEDIUM
        elif peak_score < 0.90:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.CRITICAL

    def dispatch(
        self,
        video_source: str,
        timeline_scores: List[float],
        keyframe_indices: List[int],
        boxes_per_frame: Optional[List[List]] = None
    ) -> Optional[AlertEvent]:
        """
        Evaluate scores and dispatch an alert if threshold is breached and cooldown has elapsed.
        """
        if not timeline_scores:
            return None

        peak_score = float(max(timeline_scores))
        mean_score = float(sum(timeline_scores) / len(timeline_scores))
        is_anomaly = peak_score >= self.threshold
        severity = self.compute_severity(peak_score)
        now = time.time()

        total_boxes = sum(len(b) for b in boxes_per_frame) if boxes_per_frame else 0
        person_count = total_boxes // 2  # heuristic approximation

        event = AlertEvent(
            timestamp=now,
            video_source=str(video_source),
            peak_score=round(peak_score, 4),
            mean_score=round(mean_score, 4),
            severity=severity,
            keyframe_indices=keyframe_indices,
            timeline_scores=[round(s, 4) for s in timeline_scores],
            num_detected_objects=total_boxes,
            num_person_interactions=person_count,
            is_anomaly=is_anomaly,
            summary=f"[{severity.value}] Anomaly detected (Peak: {peak_score:.2f}) on {Path(video_source).name}"
        )

        # Dispatch if anomalous and cooldown satisfied
        if is_anomaly:
            if (now - self._last_alert_time) >= self.cooldown_seconds:
                self._last_alert_time = now
                self._notify(event)
        else:
            self._notify(event)

        return event

    def _notify(self, event: AlertEvent) -> None:
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + "\n")

        for cb in self.callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[AlertDispatcher] Callback error: {e}")


def generate_dashboard_html(
    event: AlertEvent,
    output_path: Path,
    fps: float = 30.0,
    latency_ms: float = 33.3
) -> Path:
    """
    Generate an interactive, self-contained HTML monitoring dashboard.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status_color = "#ef4444" if event.is_anomaly else "#22c55e"
    status_bg = "rgba(239, 68, 68, 0.15)" if event.is_anomaly else "rgba(34, 197, 94, 0.15)"
    status_text = "ANOMALY / AGGRESSION DETECTED" if event.is_anomaly else "NORMAL SURVEILLANCE ACTIVITY"

    labels_json = json.dumps([f"Frame {kf}" for kf in event.keyframe_indices])
    scores_json = json.dumps(event.timeline_scores)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARGUS Surveillance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .title {{
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .card-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 28px;
            font-weight: 700;
        }}
        .chart-container {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .meta-list {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <span>🛡️ ARGUS</span>
            <span style="font-size: 14px; color: var(--text-secondary); font-weight: normal;">Adaptive Relational Graph Understanding System</span>
        </div>
        <div class="badge" style="background: {status_bg}; color: {status_color}; border: 1px solid {status_color};">
            {event.severity.value} • {status_text}
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-label">Peak Anomaly Score</div>
            <div class="card-value" style="color: {status_color};">{event.peak_score:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Mean Clip Score</div>
            <div class="card-value">{event.mean_score:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Keyframes Processed</div>
            <div class="card-value">{len(event.keyframe_indices)}</div>
        </div>
        <div class="card">
            <div class="card-label">Processing Throughput</div>
            <div class="card-value" style="color: var(--accent);">{fps:.1f} <span style="font-size: 14px; color: var(--text-secondary);">FPS</span></div>
        </div>
    </div>

    <div class="chart-container">
        <h3 style="margin-top: 0; font-size: 16px; margin-bottom: 16px;">Keyframe Anomaly Timeline & Attention Modulation</h3>
        <canvas id="timelineChart" height="90"></canvas>
    </div>

    <div class="card">
        <div class="card-label">Stream & Model Metadata</div>
        <div class="meta-list">
            <div><strong>Source Stream:</strong> {event.video_source}</div>
            <div><strong>Total Detected Entities:</strong> {event.num_detected_objects} bounding boxes</div>
            <div><strong>Inference Latency:</strong> {latency_ms:.1f} ms / clip</div>
            <div><strong>Decision Verdict:</strong> <span style="color: {status_color}; font-weight: bold;">{status_text}</span></div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('timelineChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels_json},
                datasets: [
                    {{
                        label: 'Anomaly Score',
                        data: {scores_json},
                        borderColor: '{status_color}',
                        backgroundColor: '{status_bg}',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }},
                    {{
                        label: 'Decision Threshold (0.50)',
                        data: Array({len(event.timeline_scores)}).fill(0.5),
                        borderColor: '#eab308',
                        borderDash: [6, 6],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        min: 0,
                        max: 1.0,
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f8fafc' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
