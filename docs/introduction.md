# Chapter 1: Introduction, Problem Statement, and Research Hypotheses

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 1.1 Context and Operational Motivation

Modern municipal and institutional security relies on pervasive sensor networks consisting of closed-circuit television (CCTV) and Real-Time Streaming Protocol (RTSP) cameras. However, the vast majority of deployed surveillance streams remain purely forensic: feeds are recorded passively and reviewed only post-incident to reconstruct how a robbery, assault, or security breach occurred.

Deploying automated video anomaly detection (VAD) and proactive threat monitoring systems in real-world control centers faces three primary bottlenecks, collectively designated as the **Surveillance Paradox**:

1. **Human Vigilance Fatigue:** Cognitive ergonomics studies reveal that security operators miss up to 45% of critical physical security incidents after only 12 minutes of continuous observation, with detection failure rates escalating beyond 95% after 22 minutes.
2. **Compute and Bandwidth Saturation:** Over 95% of 24/7 continuous CCTV surveillance consists of static background geometry. Streaming continuous raw uncompressed 1080p video across local networks saturated bandwidth and demands expensive enterprise GPU clusters.
3. **The Relational Nature of Physical Violence:** Video anomalies and threats are not static semantic textures; they are fundamentally kinetic, physical, and relational. Two individuals standing 0.5 meters apart at a bus stop constitutes normal pedestrian behavior; two individuals converging at high relative velocity with sudden bounding box overlap, rapid arm trajectory extension, and knife brandishing constitutes physical assault. Monolithic frame-level feature representations fail to isolate multi-agent relational mechanics from background motion.

---

## 1.2 The VLM Trap and the ARGUS Contrarian Thesis

Recent computer vision research has focused on adapting Large Vision-Language Models (VLMs) and Multimodal LLMs (7B+ parameters) for surveillance reasoning. While effective for open-ended queries, this approach presents critical drawbacks for edge-based surveillance:

- **The Latency Chasm:** Physical assaults unfold in 1.5 to 3.0 seconds. A 7B+ parameter VLM operating at 1.4 to 2.1 FPS takes 4 to 8 seconds to generate an alert, by which time the incident is over.
- **The Hallucination Liability:** Autoregressive language decoders can hallucinate non-existent weapons or misinterpret benign gestures, creating severe legal liabilities.
- **The Edge Viability Tax:** Requiring 14+ GB of VRAM per stream and transmitting citizen video across external cloud APIs violates privacy mandates (e.g., GDPR, CCPA) and is economically unfeasible.

**The ARGUS Thesis:** Surveillance anomalies and physical threats are not linguistic phenomena; they are physical, kinematic, and topological phenomena. By modeling anomalies as structural interaction disruptions across an explicit, dynamic scene graph coupled with causal temporal memory, ARGUS achieves high discriminability, complete mathematical determinism, zero hallucinations, and real-time edge execution.

---

## 1.3 Formal Problem Statement

Let $\mathcal{V} = \{I_1, I_2, \dots, I_T\}$ denote an infinite incoming video stream where $I_t \in \mathbb{R}^{H \times W \times 3}$. The objective is to map $\mathcal{V}$ into an online sequence of anomaly probability scores $s_t \in [0, 1]$ and structured threat telemetry subject to four operational constraints:

1. **Strict Online Temporal Causality:** Predictions at time $t$ depend strictly on historical and current observations $\{I_\tau\}_{\tau \le t}$, with zero access to future frames ($\tau > t$).
2. **Bounded Computational Complexity:** Working memory must remain strictly $\mathcal{O}(1)$ constant with respect to stream duration $T$, preventing out-of-memory crashes during 24/7 continuous operation.
3. **Real-Time Edge Throughput:** End-to-end processing must achieve at least 25 FPS on embedded edge GPUs (e.g., NVIDIA Jetson Orin) or at least 15 FPS on commodity unaccelerated CPUs.
4. **Structural Interpretability and Threat Telemetry:** Positive alarms must surface spatial actor bounding boxes, fine-grained entity class counts (Person, Vehicle, Bag, Weapon, Other), and relational interaction descriptions explaining the trigger.

---

## 1.4 Research Questions (RQs)

- **$RQ_1$ (Temporal Redundancy):** How can static surveillance video be compressed temporally without missing low-motion anomalies or violating online streaming causality?
- **$RQ_2$ (Relational Representation):** How can physical interactions between multiple agents, weapons, and objects be modeled explicitly in graph space without incurring prohibitive graph neural network overhead?
- **$RQ_3$ (Visual Efficiency):** How can visual transformer tokens be pruned based on relational topology to eliminate background self-attention FLOPs without dropping subtle threats?
- **$RQ_4$ (Causal Streaming Memory):** How can historical temporal context across multiple video clips be maintained indefinitely in bounded $\mathcal{O}(1)$ memory without future-frame leakage?

---

## 1.5 Research Hypotheses ($H_1-H_4$)

- **$H_1$ (Adaptive Sampling Efficiency):** An event-driven controller with a hard maximum frame-gap bound of 4 will eliminate over 40% of static frame computation without degrading anomaly detection recall.
- **$H_2$ (Relational Threat Graph Superiority):** Conditioning graph attention on an expanded 5-class semantic ontology and continuous 10D relational threat edge attributes (including Person-to-Weapon and Person-to-Bag) will yield a &gt;20% absolute ROC-AUC gain over an un-modeled baseline.
- **$H_3$ (Topology-Guided Token Pruning):** Discarding the top 50% of visual patch tokens based on graph node affinity will preserve &gt;98% of dense model accuracy while achieving a $4\times$ reduction in transformer self-attention FLOPs.
- **$H_4$ (Causal Streaming Memory):** A detached 16-clip FIFO temporal queue with causal masking will maintain temporal continuity across minutes of continuous video while maintaining strictly bounded memory.
