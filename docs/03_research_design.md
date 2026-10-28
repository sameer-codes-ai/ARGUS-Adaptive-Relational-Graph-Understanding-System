# Chapter 4: Research Design and Theoretical Framework

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 4.1 Theoretical Foundation: Topological Interaction Theory

Traditional computer vision frames Video Anomaly Detection (VAD) as an appearance-outlier problem: anomalous events are hypothesized to inhabit low-density regions of an unsupervised feature manifold. While applicable to industrial defect detection, this assumption fails in human surveillance because normal activities (running to catch a train, carrying heavy luggage, dancing) frequently occupy diverse visual manifolds.

ARGUS introduces **Topological Interaction Theory** for surveillance:
$$\text{Anomaly}(\mathcal{V}) \triangleq \mathcal{F}\Big(\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t), \, \mathcal{M}_{<t}\Big)$$
where:
1. $\mathcal{V}_t$ represents the discrete physical entities (Persons, Vehicles, Bags, Weapons, Objects) grounded with kinematic and visual descriptors.
2. $\mathcal{E}_t$ represents the continuous physical relational edges (spatial distance, IoU collision, relative convergence velocity, interaction type).
3. $\mathcal{M}_{<t}$ represents causal temporal memory.

An anomaly is fundamentally defined as a **topological disruption**—a sudden, non-smooth phase transition in the relational graph structure (e.g., sharp relative velocity spikes between converging actors followed by sustained bounding box overlap and weapon engagement).

---

## 4.2 Decoupled Dual-Layer Engineering Paradigm

To guarantee real-time throughput on edge hardware without sacrificing relational modeling depth, ARGUS separates operational stream ingestion from core neural reasoning into two decoupled layers:

```
[ Incoming Video Stream (RTSP / MP4 / AVI) ]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTER LAYER: Operational Infrastructure & Semantic Extraction│
│  1. Binary-Safe Stream Ingestion & Byte-Range Seeking       │
│  2. Spatial Normalization (128x128) & Native Decoding       │
│  3. Spatial Feature Backbone (ConvNeXt-Tiny / Native)        │
│  4. 5-Class Threat Detector (YOLO11n + COCO Mapping)         │
│  5. Pure-PyTorch Bilinear RoI Appearance Extractor          │
│  6. Multi-Modal Explainability Engine & Telemetry Counters  │
│  7. 5-Tier Hysteresis Alert Dispatcher                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Keyframes, Multi-scale Features,
                           │  Boxes, Semantic Classes, RoIs)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ INNER LAYER: High-Throughput Core Neural Reasoning Engine    │
│  1. Adaptive Temporal Sampler (4 Physical Motion Heuristics)│
│     * Hard Coverage Bound (G_max = 4)                       │
│     * Dynamic Attention Modulation Factor (mu_t)            │
│  2. Dynamic Threat Scene Graph Construction (12D / 10D)     │
│  3. Vectorized EdgeGAT (Scatter-Based Message Passing)      │
│  4. Topology-Aware Gated Cross-Attention Fusion             │
│  5. Graph-Guided Saliency Token Pruner (rho = 0.50)         │
│  6. Causal Streaming Temporal Transformer (Upper-Tri Mask)  │
│  7. Detached FIFO Memory Bank (S = 16 Clips, O(1) RAM)      │
│  8. Multi-Objective Scoring Head & Composite Loss           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
[ Continuous Frame Anomaly Probability s_t, Bounding Boxes, Threat Telemetry ]
```

---

## 4.3 Formal Mathematical Specifications

### 4.3.1 Node Space ($\mathbb{R}^{76}$)
Each entity node $i \in \{1, \dots, N_t\}$ is defined as:
$$\tilde{v}_i = [v_i \,\|\, f_i^{\text{RoI}}] \in \mathbb{R}^{12 + 64 = 76}$$
where:
- Kinematic Sub-vector ($6\text{D}$): $[c_x, c_y, w, h, \dot{c}_x, \dot{c}_y]$
  - Centroids: $c_x = \frac{x_1 + x_2}{2W}, \quad c_y = \frac{y_1 + y_2}{2H}$
  - Spatial Scales: $w = \frac{x_2 - x_1}{W}, \quad h = \frac{y_2 - y_1}{H}$
  - Instantaneous Velocity: $\dot{c}_x = c_x^{(t)} - c_x^{(t-1)}, \quad \dot{c}_y = c_y^{(t)} - c_y^{(t-1)}$
- 5-Class Semantic One-Hot ($5\text{D}$): $[\mathbf{1}_{\text{person}}, \mathbf{1}_{\text{veh}}, \mathbf{1}_{\text{bag}}, \mathbf{1}_{\text{weap}}, \mathbf{1}_{\text{other}}]$
- Detection Confidence ($1\text{D}$): $\text{conf}_i \in [0, 1]$
- Bilinear RoI Visual Feature ($64\text{D}$): Extracted via bilinear grid sampling on ConvNeXt multi-scale feature maps.

### 4.3.2 Relational Threat Edge Space ($\mathbb{R}^{10}$)
Directed edges are formed between entities $i, j$ with $d_{ij} < 0.42$:
$$e_{ij} = [d_{ij}, \text{IoU}_{ij}, \Delta v_{ij}, p_{ij}, \mathbf{1}_{\text{P2P}}, \mathbf{1}_{\text{P2O}}, \mathbf{1}_{\text{O2O}}, \mathbf{1}_{\text{P2W}}, \mathbf{1}_{\text{P2B}}, \mathbf{1}_{\text{W2O}}]^T \in \mathbb{R}^{10}$$
where:
- Normalized Distance: $d_{ij} = \sqrt{(c_{x,i} - c_{x,j})^2 + (c_{y,i} - c_{y,j})^2}$
- Bounding Box IoU: $\text{IoU}_{ij} = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i \cup B_j)}$
- Relative Kinetic Velocity: $\Delta v_{ij} = \sqrt{(\dot{c}_{x,i} - \dot{c}_{x,j})^2 + (\dot{c}_{y,i} - \dot{c}_{y,j})^2}$
- Interaction Persistence: $p_{ij} = \min\left(1.0, \frac{\text{consecutive\_frames}_{ij}}{16}\right)$
- Topological & Threat Flags:
  $$\mathbf{1}_{\text{P2P}} = \mathbf{1}_{\{C_i=0 \land C_j=0\}}, \quad \mathbf{1}_{\text{P2O}} = \mathbf{1}_{\{C_i=0 \land C_j \in \{1,2,3,4\}\}}$$
  $$\mathbf{1}_{\text{O2O}} = \mathbf{1}_{\{C_i \ne 0 \land C_j \ne 0\}}, \quad \mathbf{1}_{\text{P2W}} = \mathbf{1}_{\{(C_i=0 \land C_j=3) \lor (C_i=3 \land C_j=0)\}}$$
  $$\mathbf{1}_{\text{P2B}} = \mathbf{1}_{\{(C_i=0 \land C_j=2) \lor (C_i=2 \land C_j=0)\}}, \quad \mathbf{1}_{\text{W2O}} = \mathbf{1}_{\{(C_i=3 \land C_j \in \{1,4\}) \lor (C_i \in \{1,4\} \land C_j=3)\}}$$

---

## 4.4 Ablation Experimental Design

To isolate the marginal contribution of each architectural innovation, ARGUS was structured across six planned experimental ablations:

1. **Ablation 1 (Backbone Alignment):** Inner Baseline (Untuned) vs. Outer Frozen vs. Joint Fine-Tuned.
2. **Ablation 2 (Semantic Schema):** Legacy-3 Schema (Person, Vehicle, Object; 10D/7D) vs. Threat-5 Schema (Person, Vehicle, Bag, Weapon, Other; 12D/10D).
3. **Ablation 3 (Visual Grounding):** Kinematics-only vs. Heuristic Energy Patches vs. Bilinear RoI sampling.
4. **Ablation 4 (Token Pruning Ratio):** $\rho \in \{0.25, 0.50, 0.75, 1.00\}$.
5. **Ablation 5 (Sampling Bound):** Uniform stride vs. Adaptive controller with $G_{\max} \in \{2, 4, 8, \infty\}$.
6. **Ablation 6 (Temporal Memory Capacity):** Memory buffer size $S \in \{0, 4, 8, 16, 32\}$.
