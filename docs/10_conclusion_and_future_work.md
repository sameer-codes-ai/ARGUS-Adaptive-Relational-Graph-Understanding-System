# Chapter 10: Conclusion, Ethical Governance, and Future Work

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 10.1 Synthesis of Research Findings

This investigation introduced **ARGUS Threat-5**, an edge-deployable, causal graph-transformer neural framework that resolves the core dilemma of the Surveillance Paradox: reconciling high-capacity relational reasoning with low-latency edge deployment.

### Key Conclusions:
1. **Relational Scene Graphs Outperform Monolithic 3D Grids:** By transforming video frames into dynamic 5-class scene graphs with 10D continuous relational threat edge conditioning, ARGUS achieved a **0.9818 ROC-AUC** and **0.9290 F1-score** on the Real-Life Violence Dataset (RLVD). This represents a **+44.02% absolute gain** over an un-modeled baseline ($0.5416$) and outstrips 7B+ parameter Vision-Language Models while using only ~28M parameters.
2. **Deterministic Real-Time Edge Viability:** With an event-driven adaptive temporal sampler ($G_{\max} = 4$) and topology-guided token pruning ($\rho = 0.50$), ARGUS reduces processed frame volume by 48.7% and transformer self-attention FLOPs by $4\times$. It executes at **31.2 FPS on commodity unaccelerated CPUs** and over **105 FPS on edge GPUs**.
3. **Strict Online Temporal Causality:** Enforcing upper-triangular causal attention masking and a detached 16-clip FIFO memory bank ensures strictly constant $\mathcal{O}(1)$ RAM usage over continuous 24/7 video streams without memory-leak crashes.
4. **Zero-Shot Cross-Dataset Generalization:** Transferring the model zero-shot to UCF-Crime yielded an **80.00% anomaly recall**, demonstrating that physical relational kinetics generalize across divergent camera perspectives and scene geographies.

---

## 10.2 Ethical AI Governance and Privacy-by-Design

Deploying AI surveillance in public spaces demands rigorous ethical oversight and legal compliance (e.g., EU AI Act, GDPR, CCPA):

### 1. Zero Raw Video Exfiltration
Raw video pixel buffers exist strictly within volatile RAM on the local edge appliance and are immediately overwritten upon processing. Only anonymous JSON metadata and vector embeddings traverse network infrastructure.

### 2. Elimination of Demographic and Biometric Profiling
ARGUS explicitly rejects facial recognition, demographic profiling, emotion analysis, and gait tracking. Graph entity nodes encode anonymous kinematic bounding box coordinates, velocities, and broad object types. Classification depends strictly on physical actions, relative velocity, and weapon possession rather than personal identity.

### 3. Human-in-the-Loop Decision Support
ARGUS is engineered strictly as a **Tier-1 Decision Support Tool**. It is neither designed nor authorized to execute autonomous punitive or physical security interventions. All critical alarms require human visual verification via the explainability cyber-HUD prior to dispatching security personnel.

---

## 10.3 Technical Limitations

1. **Low-Motion Property Crimes:** Subtle, stealthy anomalies (such as pickpocketing or quiet shoplifting) exhibit minimal relative velocity and low bounding box overlap, making them harder to isolate than physical altercations without specialized micro-motion feature heads.
2. **Severe Night-Vision Noise:** Infrared camera feeds in zero-lux environments exhibit high gain noise and severe contrast loss, occasionally causing missed bounding box detections for distant actors.
3. **Handheld Camera Dynamic Bias:** Models trained on mobile video (RLVD) learn motion priors that can trigger false alarms on sudden CCTV pan-tilt-zoom adjustments unless normalized.

---

## 10.4 Three-Horizon Future Research Roadmap

```
Horizon 1: 2027 — Edge Hardening & Micro-Motion
 ├── TensorRT INT8 Quantization (>120 FPS on Jetson Orin under 15W)
 ├── Focal Multiple Instance Learning for low-motion property crimes
 └── Resolution Scaling (256x256) for distant small-object detection

Horizon 2: 2028 — Multimodal Acoustic-Visual Fusion
 ├── Audio Spectrogram Transformer (AST) Integration
 ├── Edge-binding of gunshots, breaking glass, and screams to spatial nodes
 └── Cross-modal acoustic-relational verification

Horizon 3: 2029–2030 — City-Scale Multi-Camera Continuous Modeling
 ├── Continuous-Time Graph Neural ODEs for trajectory modeling
 ├── Multi-Camera Re-Identification without biometric storage
 └── City-wide spatio-temporal causal threat graphs
```

---

## 10.5 Concluding Remarks

ARGUS Threat-5 demonstrates that physical relational inductive biases, when coupled with causal temporal memory and topology-guided pruning, offer a superior paradigm to monolithic deep networks and massive black-box language models for life-critical physical surveillance. By delivering unmatched detection accuracy, determinism, explainability, and edge throughput, ARGUS paves the way for responsible, proactive automated surveillance.
