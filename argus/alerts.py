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

    status_color = "#f43f5e" if event.is_anomaly else "#10b981"
    status_bg = "rgba(244, 63, 94, 0.12)" if event.is_anomaly else "rgba(16, 185, 129, 0.12)"
    status_border = "rgba(244, 63, 94, 0.28)" if event.is_anomaly else "rgba(16, 185, 129, 0.28)"
    status_text = "ANOMALOUS INCIDENT DETECTED" if event.is_anomaly else "NOMINAL SURVEILLANCE ACTIVITY"

    labels_json = json.dumps([f"KF {kf}" for kf in event.keyframe_indices])
    scores_json = json.dumps(event.timeline_scores)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARGUS Surveillance Incident Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-base: #080b11;
            --bg-surface: #121824;
            --bg-subtle: #0d121c;
            --border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-tertiary: #64748b;
            --accent: #38bdf8;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            margin: 0;
            padding: 28px;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }}
        .title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand-badge {{
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.2px;
            color: var(--text-primary);
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-family: 'JetBrains Mono', monospace;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}
        .card {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }}
        .card-label {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .chart-container {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .meta-list {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.8;
            font-family: 'Inter', sans-serif;
        }}
        .meta-list strong {{
            color: var(--text-primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <div>
                    <div class="brand-badge">ARGUS &bull; Surveillance Incident Report</div>
                    <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 2px;">Causal Relational Graph Anomaly Detection</div>
                </div>
            </div>
            <div class="badge" style="background: {status_bg}; color: {status_color}; border: 1px solid {status_border};">
                {event.severity.value} &bull; {status_text}
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
                <div class="card-label">Inference Throughput</div>
                <div class="card-value" style="color: var(--accent);">{fps:.1f} <span style="font-size: 12px; color: var(--text-tertiary);">FPS</span></div>
            </div>
        </div>

        <div class="chart-container">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                <span style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-tertiary)">Causal Anomaly Probability Curve</span>
                <span style="font-size:11px;color:var(--text-tertiary);font-family:'JetBrains Mono',monospace">Decision Threshold: 0.50</span>
            </div>
            <canvas id="timelineChart" height="80"></canvas>
        </div>

        <div class="card">
            <div class="card-label">Stream Telemetry & Metadata</div>
            <div class="meta-list">
                <div><strong>Source Stream:</strong> {event.video_source}</div>
                <div><strong>Detected Entities:</strong> {event.num_detected_objects} bounding boxes</div>
                <div><strong>Inference Latency:</strong> {latency_ms:.1f} ms / clip</div>
                <div><strong>Classification Verdict:</strong> <span style="color: {status_color}; font-weight: 600;">{status_text}</span></div>
            </div>
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
                        borderWidth: 2,
                        fill: true,
                        tension: 0.25,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    }},
                    {{
                        label: 'Decision Threshold (0.50)',
                        data: Array({len(event.timeline_scores)}).fill(0.5),
                        borderColor: '#f59e0b',
                        borderDash: [4, 4],
                        borderWidth: 1.5,
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
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#64748b', font: {{ family: "'JetBrains Mono', monospace", size: 9 }} }}
                    }},
                    x: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#64748b', font: {{ family: "'JetBrains Mono', monospace", size: 9 }} }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8', font: {{ family: "'Inter', sans-serif", size: 10 }} }} }}
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
