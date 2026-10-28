# Chapter 3: Scientific Novelty and Theoretical Contributions

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 3.1 Overview of Scientific Contributions

ARGUS Threat-5 advances beyond conventional Video Anomaly Detection (VAD) by formulating surveillance monitoring as a **physical topological disruption problem** across an explicit, dynamic scene graph. Table 3.1 categorizes the scientific novelty across all structural modules.

### Table 3.1: Detailed Scientific Novelty Matrix

| Module | Prior Practice | ARGUS Threat-5 Formulation | Novelty Tier | Primary Benefit |
| :--- | :--- | :--- | :---: | :--- |
| **1. Adaptive Temporal Sampling** | Uniform frame stride or complex RL policy | 4-heuristic physical motion controller with hard coverage bound ($G_{\max}=4$) and modulation factor $\mu_t$ | **Tier 2 (Methodological)** | 48.7% frame reduction; zero temporal blind spots |
| **2. Spatial Backbone** | Heavy 3D CNNs (I3D, C3D) or full ViT | ConvNeXt-Tiny with NativeConvNeXt pure-PyTorch fallback; multi-scale feature taps | **Tier 3 (Engineering)** | High throughput on unaccelerated CPUs |
| **3. 5-Class Threat Scene Graph** | Unlabeled bounding box centroids (4D/6D) | 12D kinematic-semantic node vectors (5 classes: Person, Vehicle, Bag, Weapon, Other) + 64D bilinear RoI appearance features (76D total) | **Tier 1 (High Novelty)** | Explicit weapon grounding and actor state tracking |
| **4. 10D Relational Threat Edges** | Binary spatial distance threshold | 10D continuous vectors: 4 kinematic + 3 broad (P2P, P2O, O2O) + 3 threat flags (P2W, P2B, W2O) | **Tier 1 (High Novelty)** | Explicit isolation of weapon pulling, brawling, and bag snatching |
| **5. Vectorized EdgeGAT** | Standard GCN / GAT requiring external C++ PyG/DGL libraries | Pure-PyTorch edge-conditioned multi-head graph attention using native scatter primitives | **Tier 1 (High Novelty)** | Executes in &lt;1.5 ms/frame on standard CPU with zero C++ dependencies |
| **6. Gated Cross-Attention Fusion** | Concatenation or quadratic visual-graph cross-attention | Asymmetric gated residual cross-attention (Visual Patches query Graph Nodes) with learned sigmoid gating | **Tier 2 (Methodological)** | Graph context injected directly into patch tokens |
| **7. Topology-Guided Token Pruning** | Unsupervised visual similarity clustering (ToMe, DynamicViT) | Cross-modal graph-affinity saliency scoring: top 50% retained ($\rho = 0.50$) | **Tier 1 (High Novelty)** | $4\times$ self-attention FLOP reduction; preserves 99.96% dense AUC |
| **8. Causal Streaming Temporal Transformer**| Offline bidirectional clip self-attention | Strict upper-triangular causal masking; dynamic attention modulation $\mu_t$ | **Tier 2 (Methodological)** | True online streaming without future-frame leakage |
| **9. Detached FIFO Memory Bank** | Monolithic sequence input or unbounded RNN hidden states | Fixed-capacity 16-clip circular memory queue with detached gradients | **Tier 2 (Methodological)** | Strictly $\mathcal{O}(1)$ constant memory; prevents 24/7 crash leaks |
| **10. Multi-Objective Loss & Telemetry** | Isolated BCE loss or isolated MIL ranking | Composite loss: BCE + smoothness + sparsity + MIL ranking; real-time 5-tier hysteresis alerting with object telemetry | **Tier 1 (High Novelty)** | Suppresses score flickering; emits concurrent entity counts (`person_count`, `weapon_count`) |

---

## 3.2 Detailed Analysis of Key Tier-1 Novelties

### 3.2.1 Expanded 5-Class Threat Scene Graph Ontology
Prior graph-based surveillance works either utilize class-agnostic bounding box centroids or simple person-only graphs. ARGUS Threat-5 defines a formal 5-class semantic taxonomy:
- **Class 0 (Person):** Primary active agents.
- **Class 1 (Vehicle):** Bicycles, cars, motorcycles, trucks, buses.
- **Class 2 (Bag):** Backpacks, handbags, suitcases (critical for unattended baggage and theft).
- **Class 3 (Weapon):** Knives, firearms, bladed weapons (mapped from COCO class 43 and domain detections).
- **Class 4 (Other Object):** Barriers, furniture, ambient objects.

By pairing each entity with a 12-dimensional kinematic-semantic vector and 64-dimensional bilinear RoI visual features sampled directly from ConvNeXt feature maps, ARGUS resolves the spatial grounding ambiguity that plagues monolithic 3D CNNs.

### 3.2.2 Continuous 10-Dimensional Relational Threat Edge Attributes
Unlike standard Graph Attention Networks that evaluate scalar edge weights, ARGUS modulates message passing across 10 continuous and categorical attributes:
$$e_{ij} = \big[\text{dist}_{ij}, \text{IoU}_{ij}, \Delta v_{ij}, \text{persistence}_{ij}, \mathbf{1}_{\text{P2P}}, \mathbf{1}_{\text{P2O}}, \mathbf{1}_{\text{O2O}}, \mathbf{1}_{\text{P2W}}, \mathbf{1}_{\text{P2B}}, \mathbf{1}_{\text{W2O}}\big]^T$$

This explicit conditioning enables the network to differentiate:
1. **Benign Convergence vs. Physical Violence:** Two people walking side-by-side maintain low $\Delta v_{ij}$ and zero $\text{IoU}_{ij}$. During an assault, relative velocity spikes and $\text{IoU}_{ij}$ exceeds 0.3 with active P2P flags.
2. **Benign Tool Usage vs. Weapon Brandishing:** When a person approaches another with an active weapon, the P2W edge flag dramatically amplifies attention weights in EdgeGAT.

### 3.2.3 Vectorized Pure-PyTorch EdgeGAT
Deploying Graph Neural Networks on edge hardware typically requires complex C++ compiled libraries (e.g., PyTorch Geometric, DGL). ARGUS implements EdgeGAT entirely using PyTorch tensor scatter primitives (`torch.scatter_reduce_`, `torch.scatter_add_`), achieving full multi-head edge conditioning in under 1.5 ms per frame on commodity x86-64 and ARM edge CPUs.

### 3.2.4 Topology-Guided Saliency Token Pruning
Vision Transformers typically compute self-attention across all visual patch tokens ($N_p = 64$ for $128 \times 128$ images). In surveillance feeds, over 70% of patches represent invariant background structures (floors, brick walls, ceiling tiles). Rather than relying on unsupervised image clustering, ARGUS evaluates patch affinity against active scene graph nodes. Patches overlapping interacting actors or weapons are prioritized, while static background patches are pruned, slashing transformer FLOPs by $4\times$.

---

## 3.3 Historical Failure Modes and Threat-5 Resolutions

### 1. The False-Negative Collapse of Frozen Backbones
- *Identified Flaw:* Evaluating an ImageNet-pretrained ConvNeXt backbone with frozen weights caused a catastrophic recall drop to $5.63\%$ ($F_1 = 0.1026$), because ImageNet features represent clean, centered objects rather than low-resolution, motion-blurred CCTV interactions.
- *Threat-5 Resolution:* Joint end-to-end fine-tuning with differential learning rates ($1 \times 10^{-5}$ on backbone, $1 \times 10^{-4}$ on heads) aligned low-level filters with surveillance kinetics, lifting recall to $96.00\%$ on RLVD ($F_1 = 0.9290$).

### 2. The Weapon Misclassification Blind Spot
- *Identified Flaw:* The 3-class schema (Person, Vehicle, Generic Object) lumped weapons into generic objects. Small knives and firearms were drowned out by large background objects (desks, benches).
- *Threat-5 Resolution:* Threat-5 created a distinct Weapon class (Class 3) and conditioned edges with explicit Person-to-Weapon (P2W) and Weapon-to-Other (W2O) indicators, lifting in-domain AUC from $0.8845$ to $0.9818$ (+9.73%).
