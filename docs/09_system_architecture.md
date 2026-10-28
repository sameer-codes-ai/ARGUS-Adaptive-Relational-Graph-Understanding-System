# Chapter 9: System Architecture, Deployment Topologies, and Operational Alerting

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 9.1 Enterprise Deployment Topologies

ARGUS Threat-5 is engineered for flexible deployment across three production environments:

```
Topology A: Decentralized Edge Micro-Gateway
┌────────────────┐     RTSP / CSI     ┌────────────────────────┐     MQTT / Webhook     ┌──────────────────────┐
│ IP Camera Feeds│ ─────────────────> │ NVIDIA Jetson Orin     │ ─────────────────────> │ Central Sec. Console │
│ (1080p, 25 FPS)│                    │ (ARGUS Threat-5 Engine)│                        │ (Alerts & Telemetry) │
└────────────────┘                    └────────────────────────┘                        └──────────────────────┘

Topology B: Centralized On-Premises GPU Server
┌────────────────┐                                                                      ┌──────────────────────┐
│ Camera Cluster │ ─────────┐                                                    ┌────> │ Security Operations  │
└────────────────┘          │                                                    │      │ Center (SOC) Console │
┌────────────────┐          │ RTSP Multi-Stream ┌──────────────────────────────┐ │      └──────────────────────┘
│ Camera Cluster │ ─────────┼─────────────────> │ Multi-GPU Enterprise Server  │─┤
└────────────────┘          │                   │ (ARGUS 8-Stream Parallel Hub)│ │      ┌──────────────────────┐
┌────────────────┐          │                   └──────────────────────────────┘ └────> │ Evidence Store / NAS │
│ Camera Cluster │ ─────────┘                                                           │ (Zero Raw Video)     │
└────────────────┘                                                                      └──────────────────────┘

Topology C: Hybrid Edge-Filter Hierarchy
┌────────────────┐     Raw Stream     ┌────────────────────────┐     Keyframes Only     ┌──────────────────────┐
│ CCTV Hardware  │ ─────────────────> │ Lightweight Edge Filter│ ─────────────────────> │ Central Neural Hub   │
│                │                    │ (Module 1 Sampler)     │                        │ (Full ARGUS Pipeline)│
└────────────────┘                    └────────────────────────┘                        └──────────────────────┘
```

1. **Decentralized Edge Micro-Gateway (Topology A):**
   - Deployed directly on edge hardware (e.g., NVIDIA Jetson Orin Nano / AGX) located at camera mount points.
   - Operates in FP16/INT8 precision at >105 FPS, consuming under 15 Watts.
   - Raw video never leaves the device; only lightweight JSON telemetry packets are transmitted over municipal networks.
2. **Centralized On-Premises GPU Server (Topology B):**
   - Ingests multiple RTSP camera feeds (e.g., 8–16 cameras on an RTX 4090).
   - Serves an enterprise Security Operations Center (SOC) dashboard with live timeline scrubbing and sub-50 ms alert latency.
3. **Hybrid Edge-Filter Hierarchy (Topology C):**
   - Module 1 (Adaptive Temporal Sampler) runs on low-cost microcontrollers (e.g., Raspberry Pi or camera DSPs), dropping 48.7% of static background frames locally.
   - Keyframe bursts are routed to a central server executing Modules 2 through 9.

---

## 9.2 Calibrated 5-Tier Hysteresis Alert Dispatcher

To eliminate operator alert fatigue while ensuring guaranteed notification for life-threatening emergencies, ARGUS implements a stateful Alert Dispatcher with hysteresis:

### Table 9.1: Calibrated Operational Alert Hierarchy

| Alert Tier | Score Range | Operational Meaning | Action Triggered | Cooldown Period |
| :--- | :---: | :--- | :--- | :---: |
| **NORMAL** | $s < 0.35$ | Routine pedestrian traffic, ambient flow | Silent logging | None |
| **LOW** | $0.35 \le s < 0.50$ | Fast convergence, running, grouping | Background log, flag clip | None |
| **MEDIUM** | $0.50 \le s < 0.70$ | Minor confrontation, shoving, loitering | Audit alert, chime | 3.0 seconds |
| **HIGH** | $0.70 \le s < 0.85$ | Escalating altercation, physical struggle | Priority alarm, popup HUD | 5.0 seconds |
| **CRITICAL** | $s \ge 0.85$ | Confirmed assault, weapon brandishing | High-priority security dispatch | 5.0 seconds |

### Hysteresis and Refractory Logic:
If an altercation persists across several seconds, sending an alert on every frame would overwhelm monitoring staff. When an alert of tier $\ge \text{MEDIUM}$ is emitted, a refractory cooldown timer $\tau_{\text{cooldown}} = 5.0\text{ seconds}$ is engaged. Subsequent frames within this window update the telemetry payload without triggering duplicate audible alarms or webhook dispatches.

---

## 9.3 Structured JSON Threat Telemetry Payload

Each alert event dispatches a structured JSON payload to the central security broker:

```json
{
  "event_id": "evt_20261028_213315_cam04",
  "camera_id": "CAM_WEST_GATE_04",
  "timestamp": "2026-10-28T21:33:15.420Z",
  "anomaly_score": 0.942,
  "severity_tier": "CRITICAL",
  "telemetry": {
    "person_count": 3,
    "weapon_count": 1,
    "bag_count": 0,
    "vehicle_count": 0,
    "other_count": 1
  },
  "spatial_attributions": [
    {
      "entity_id": 0,
      "class": "person",
      "bbox": [0.34, 0.45, 0.52, 0.88],
      "confidence": 0.92
    },
    {
      "entity_id": 1,
      "class": "weapon",
      "bbox": [0.48, 0.62, 0.55, 0.71],
      "confidence": 0.88
    }
  ],
  "dominant_interaction": "Person-to-Weapon (P2W) Engagement & High-Velocity Strike"
}
```

---

## 9.4 Production Web Console and Cyber-HUD

The production web interface (`web_demo.py` and `frontend.html`) features:
1. **Dual-Mode Video Player:** Standard 30 FPS playback synchronized with an AI Cyber-HUD keyframe overlay rendering actor bounding boxes, weapon badges, and EdgeGAT relational lines.
2. **Interactive Chart.js Anomaly Timeline:** Displays causal frame-by-frame anomaly probability scores with color-coded severity regions.
3. **Dynamic Sensitivity Slider:** Permits operators to adjust the anomaly threshold ($0.10$ to $0.85$) dynamically with instant client-side recalculation.
4. **Standalone Portability:** Built with zero npm or node dependencies, running entirely within a single Python script.
