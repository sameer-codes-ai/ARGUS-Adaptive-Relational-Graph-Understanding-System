# ARGUS: Presentation Slide Deck (15 Slides)
## Adaptive Relational Graph Understanding for Surveillance — Threat-5 Release
**Authors:** Omkar Joshi, Sameer Kumar  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore  
**Guide / Review:** Academic Project Review & Defense  

---

### Slide 1: Title & Overview
- **Header:** ARGUS: Adaptive Relational Graph Understanding for Surveillance
- **Sub-header:** An Edge-Viable Causal Graph-Transformer Framework with Expanded 5-Class Threat Semantics
- **Presenter Information:** Omkar Joshi & Sameer Kumar | School of CSE, VIT Vellore
- **Key Badges:** 
  - **0.9818 ROC-AUC** (Threat-5 RLVD Champion Checkpoint)
  - **92.67% Accuracy | 0.9290 F1-Score | 96.00% Recall**
  - **31.2 FPS on Commodity CPU** (>105 FPS on Edge GPU)
  - **Strict Online Temporal Causality** ($G_{\max} = 4, \mathcal{O}(1)$ Memory)
- **Visual Suggestion:** Dual-pane split: on the left, ARGUS logo and institutional header; on the right, screenshot of the ARGUS Cyber-HUD surveillance player with actor bounding boxes, weapon indicator badges, and EdgeGAT relational lines.
- **Speaker Notes:** "Good morning, respected professors. Today we present ARGUS Threat-5, an edge-deployable, causal graph-transformer neural framework for real-time video anomaly detection and violent threat recognition in surveillance networks."

---

### Slide 2: The Surveillance Paradox & Motivation
- **Header:** The Problem: Why Current Surveillance Systems Fail
- **Core Dilemma:** The *Surveillance Paradox* — billions of CCTV cameras exist worldwide, yet 99% of surveillance remains purely forensic (recorded passively, reviewed only after a crime occurs).
- **Three Core Operational Bottlenecks:**
  1. **Human Vigilance Fatigue:** Operators miss up to 45% of critical incidents after 12 minutes, and up to 95% after 22 minutes.
  2. **Compute & Bandwidth Saturation:** Over 95% of 24/7 CCTV video is empty, static background. Streaming raw uncompressed video across networks is unsustainable.
  3. **The Relational Nature of Physical Violence:** An altercation is not a static texture. Two people walking together is normal; two people converging at high velocity with physical contact and weapon brandishing is an assault. Monolithic pixel grids fail to isolate these multi-agent relational dynamics.
- **Speaker Notes:** "Human operators cannot watch dozens of monitors continuously. Meanwhile, traditional deep learning models waste 95% of their compute on empty walls and static pavements without understanding physical relationships."

---

### Slide 3: The Failure of Existing Paradigms
- **Header:** Limitations of Prior Art: 3D CNNs vs. Large VLMs
- **Comparison Table:**
  | Dimension | 3D CNNs / VideoMAE | Large VLMs (7B+ Params) | ARGUS Threat-5 (Our Approach) |
  |---|---|---|---|
  | **Representation** | Monolithic 3D pixel grid | Autoregressive text tokens | **5-Class Dynamic Scene Graph** |
  | **Compute Footprint** | Heavy 3D convolutions | Prohibitive (>14 GB VRAM) | **Lightweight (~28M Params)** |
  | **Latency** | 150–300 ms per clip | 4–8 seconds (1.4 FPS) | **32 ms / 31.2 FPS on CPU** |
  | **Operational Risk** | Black-box opacity | Generative hallucinations | **Deterministic & Explainable** |
  | **Edge Viability** | Requires server GPU | Impossible on edge CCTV | **Pure PyTorch CPU/NPU/Edge GPU** |
- **The VLM Trap:** A physical assault lasts 1.5–3.0 seconds. A 7B parameter VLM running at 1.5 FPS takes 6 seconds to respond — by then, the incident is over.
- **Speaker Notes:** "While modern research has chased 7-billion parameter vision-language models, they are unusable on edge cameras due to latency and hallucinations. ARGUS treats anomalies not as language, but as physical topological disruptions."

---

### Slide 4: End-to-End System Architecture (Dual-Layer Topology)
- **Header:** System Topology: Decoupled Dual-Layer Engine
- **Outer Layer (Operational Infrastructure):**
  - Binary-Safe Multipart Stream Ingestion (HTTP 206 Byte-Range Seeking)
  - Fine-Tuned ConvNeXt-Tiny Spatial Feature Backbone
  - 5-Class Threat Detector (YOLO11n + COCO Mapping) & Bilinear RoI Extractor
  - Multi-Tier Hysteresis Alert Dispatcher & Live Web Console
- **Inner Layer (High-Throughput Core Neural Engine):**
  - Event-Driven Adaptive Temporal Sampler ($G_{\max} = 4$)
  - Vectorized EdgeGAT (Edge-Conditioned Graph Attention Network)
  - Topology-Aware Gated Cross-Attention Fusion
  - Graph-Guided Visual Token Pruner (50% token reduction $\rightarrow 4\times$ FLOP savings)
  - Causal Temporal Transformer with 16-Clip Streaming FIFO Memory
- **Speaker Notes:** "ARGUS decouples operational video handling from the neural reasoning engine, allowing sub-2-second inference on standard commodity hardware."

---

### Slide 5: Step 1 — Video Ingestion & Binary Stream Parsing
- **Header:** Step 1: Ingestion, Decoding & Byte-Range Setup
- **What Happens When a Video is Uploaded:**
  1. **Binary-Safe Multipart Stream Ingestion:** Multi-threaded server parses incoming binary payload without MIME corruption (handles MP4/AVI/MOV up to 100MB).
  2. **Metadata & Property Extraction:** OpenCV reads native framerate (e.g., 25/30 FPS), total frame count, aspect ratio, and video duration.
  3. **HTTP 206 Partial Content Preparation:** Enables byte-range streaming (`Range: bytes=start-end`), allowing zero-latency timeline seeking on the web client.
  4. **Standardized Spatial Scaling:** Decodes frames and normalizes tensor dimensions to $128 \times 128$ for optimal edge throughput.
- **Speaker Notes:** "The pipeline begins the instant a video file or live stream is received. We immediately extract physical video metadata and set up partial-content streaming so the user experiences zero buffering."

---

### Slide 6: Step 2 — Event-Driven Adaptive Temporal Sampling
- **Header:** Step 2: Adaptive Sampling (Filtering 48.7% Redundant Frames)
- **The Mathematical Formulation:**
  - Evaluates inter-frame kinetic changes using four lightweight indicators:
    $$\text{Pixel Difference: } D_t = \frac{1}{HW}\sum |I_t - I_{t-1}|$$
    $$\text{Spatial Gradient Energy: } G_t = \frac{1}{HW}\sum (|\nabla_x I_t| + |\nabla_y I_t|)$$
    $$\text{Shannon Entropy: } H_t = -\sum p_b \log_2(p_b + \epsilon)$$
    $$\text{Scene Cut Divergence: } S_t = 1 - \frac{\sum \min(p_t, p_{t-1})}{\sum \max(p_t, p_{t-1})}$$
  - Composite motion metric: $M_t \ge \tau_s = 0.12$
- **Hard Coverage Bound ($G_{\max} = 4$):**
  - If 4 consecutive frames fall below threshold $\tau_s$, a frame is forcibly ingested.
  - *Guarantees creeping intrusions and slow-moving anomalies are never missed.*
- **Outcome:** **48.7% frame reduction** without sacrificing anomaly recall.
- **Speaker Notes:** "Instead of blindly processing all 30 frames every second, Module 1 uses motion gradients and entropy to skip empty frames, while the hard bound of 4 guarantees slow-moving events are never skipped."

---

### Slide 7: Step 3 — Spatial Backbone & 5-Class Threat Detector
- **Header:** Step 3: ConvNeXt Feature Extraction & 5-Class Threat Grounding
- **Spatial Feature Backbone:**
  - Jointly fine-tuned ConvNeXt-Tiny architecture with differential learning rates ($0.1\times$ backbone, $1.0\times$ head).
- **Expanded 5-Class Threat Ontology:**
  - Person (0), Vehicle (1), Bag (2), Weapon (3), Other Object (4).
  - COCO class 43 (knife) $\to$ Weapon; COCO classes 24, 26, 28 (backpack, handbag, suitcase) $\to$ Bag.
- **12D Kinematic-Semantic Node Vector:**
  $$v_i = [c_x, c_y, w, h, \dot{c}_x, \dot{c}_y, \mathbf{1}_{\text{person}}, \mathbf{1}_{\text{veh}}, \mathbf{1}_{\text{bag}}, \mathbf{1}_{\text{weap}}, \mathbf{1}_{\text{other}}, \text{conf}_i]^T$$
- **64D Bilinear RoI Visual Appearance:**
  - Samples ConvNeXt feature maps directly at actor bounding boxes.
  - **Total Node Representation: 76-dimensional descriptor.**
- **Speaker Notes:** "Here we isolate individual entities across 5 explicit semantic classes. Each entity receives a 76-dimensional descriptor encoding its spatial location, velocity, threat classification, and localized visual appearance."

---

### Slide 8: Step 4 — Dynamic Relational Threat Scene Graph
- **Header:** Step 4: Modeling Threat Interactions (10D Threat Edges)
- **Constructing Graph Edges:**
  - Directed edges are instantiated between entity pairs within proximity threshold $d < 0.42$.
- **Continuous 10D Relational Threat Edge Conditioning:**
  $$e_{ij} = \big[d_{ij},\, \text{IoU}_{ij},\, \Delta v_{ij},\, p_{ij},\, \mathbf{1}_{\text{P2P}},\, \mathbf{1}_{\text{P2O}},\, \mathbf{1}_{\text{O2O}},\, \mathbf{1}_{\text{P2W}},\, \mathbf{1}_{\text{P2B}},\, \mathbf{1}_{\text{W2O}}\big]^T$$
- **Six Explicit Interaction Categories:**
  - **Person-to-Person (P2P):** Physical brawls, wrestling, strikes.
  - **Person-to-Object (P2O):** Property theft, arson, vandalism.
  - **Object-to-Object (O2O):** Vehicle crashes, barrier impacts.
  - **Person-to-Weapon (P2W):** Knife wielding, armed threatening.
  - **Person-to-Bag (P2B):** Bag snatching, unattended luggage placement.
  - **Weapon-to-Other (W2O):** Weapon strikes against objects or barriers.
- **Speaker Notes:** "This is the core differentiator of ARGUS Threat-5. We explicitly model relationships. When a person wields a weapon, the P2W edge flag activates, dramatically focusing the network on that critical threat link."

---

### Slide 9: Step 5 — Vectorized EdgeGAT (Graph Attention Network)
- **Header:** Step 5: Relational Message Passing via EdgeGAT
- **Why Standard GNNs Fail:** Standard GCNs and GATs treat edges as binary 0 or 1 links, ignoring continuous physical kinematics between actors.
- **Mathematical Attention Formulation:**
  $$\alpha_{ij} = \frac{\exp\Big(\text{LeakyReLU}\big(a^T [W_v \tilde{v}_i \,\|\, W_v \tilde{v}_j \,\|\, W_e e_{ij}]\big)\Big)}{\sum_{k \in \mathcal{N}_i} \exp\Big(\text{LeakyReLU}\big(a^T [W_v \tilde{v}_i \,\|\, W_v \tilde{v}_k \,\|\, W_e e_{ik}]\big)\Big)}$$
- **Node State Update:**
  $$h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W_v h_j\right)$$
- **High-Performance Vectorization:**
  - Built with pure PyTorch scatter operations (`scatter_reduce_`, `scatter_add_`). Zero Python loops.
  - Executes in **$<1.5\text{ ms}$ per frame** on standard CPU.
- **Speaker Notes:** "EdgeGAT modulates attention by physical edge features. If two people are colliding at high speed or a weapon is brandished, the attention weight $\alpha_{ij}$ spikes, focusing the network on that specific interaction."

---

### Slide 10: Step 6 — Topology-Aware Fusion & Token Pruning
- **Header:** Step 6: Fusing Visuals with Graphs & Pruning Backgrounds
- **The Quadratic Bottleneck:** Standard Vision Transformers scale quadratically $\mathcal{O}(N^2)$ with the number of visual patch tokens.
- **Two-Stage Efficiency Optimization:**
  1. **Gated Cross-Attention Fusion:**
     - Visual patch tokens act as Queries ($Q$).
     - Relational scene graph embeddings act as Keys ($K$) and Values ($V$).
     - Injects relational graph context into visual feature representations.
  2. **Graph-Guided Saliency Token Pruner:**
     - Computes spatial affinity between each visual patch and active graph nodes.
     - Discards the top 50% uninformative background tokens (empty walls, ceilings, sky).
- **Impact:** **50% sequence length reduction**, achieving **$4\times$ FLOP savings** while retaining **99.96%** of dense model accuracy.
- **Speaker Notes:** "By pruning background tokens that contain no human interaction, we reduce sequence length by half and cut transformer computational complexity by a factor of 4 without losing accuracy."

---

### Slide 11: Step 7 — Causal Streaming Temporal Transformer
- **Header:** Step 7: Online Causal Reasoning Across Time
- **Strict Online Temporal Causality:**
  - In live surveillance, future frames do not exist.
  - ARGUS enforces an **upper-triangular causal attention mask**: attention at time $t$ can only attend to time steps $\tau \le t$.
- **Bounded FIFO Streaming Memory ($S = 16$ Clips):**
  - Detached circular memory queue maintains temporal context across minutes of continuous video.
  - Memory footprint is strictly $\mathcal{O}(1)$ constant, preventing memory leak crashes during 24/7 continuous deployment.
- **Dynamic Attention Modulation ($\mu_t$):**
  - Modulates temporal attention logits with the sampling controller's motion density factor.
- **Speaker Notes:** "Real-time surveillance requires strict causality. Unlike models that require the whole video file before predicting, ARGUS processes continuously frame-by-frame with zero future lookahead."

---

### Slide 12: Step 8 — Multi-Objective Scoring & Losses
- **Header:** Step 8: Anomaly Probability Scoring & Optimization
- **Output:** Emits continuous frame anomaly probability $s_t \in [0, 1]$.
- **Composite 4-Term Optimization Loss:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{smooth}} + \lambda_2 \mathcal{L}_{\text{sparse}} + \lambda_3 \mathcal{L}_{\text{MIL}}$$
  - **$\mathcal{L}_{\text{BCE}}$:** Binary cross-entropy on clip anomaly state.
  - **$\mathcal{L}_{\text{smooth}} = \sum_t (s_t - s_{t-1})^2$:** Suppresses high-frequency frame flicker and noise.
  - **$\mathcal{L}_{\text{sparse}} = \frac{1}{T}\sum_t s_t$:** Enforces anomaly rarity (violent events are localized bursts).
  - **$\mathcal{L}_{\text{MIL}}$:** Multiple Instance Learning ranking loss ensuring anomalous bags score higher than normal bags.
- **Weights:** $\lambda_1 = 0.10, \lambda_2 = 0.01, \lambda_3 = 0.50$.
- **Speaker Notes:** "Our loss function ensures three properties: high classification accuracy, smooth non-flickering predictions across frames, and temporal sparsity so baseline scores stay close to zero."

---

### Slide 13: Step 9 — Operational Alert Dispatcher & Telemetry
- **Header:** Step 9: Alert Dispatching & Real-Time Telemetry
- **Eliminating False Alarms:** Normal pedestrian walking scores under $0.05$ ($<5\%$).
- **Calibrated Multi-Tier Severity Hierarchy:**
  | Tier | Score Range | Operational Meaning | Action Triggered |
  |---|:---:|---|---|
  | **NORMAL** | $< 0.35$ | Routine pedestrian flow | Silent logging |
  | **LOW** | $0.35 - 0.50$ | Suspicious clustering / fast approach | Flag for review |
  | **MEDIUM** | $0.50 - 0.70$ | Confrontation / theft | Audio alert chime |
  | **HIGH** | $0.70 - 0.85$ | Escalating violence / weapon pulled | High-priority flag |
  | **CRITICAL** | $\ge 0.85$ | Verified assault / active weapon | Security dispatch |
- **Real-Time Telemetry:** Emits concurrent object counts (`person_count`, `weapon_count`, `bag_count`, `vehicle_count`) with every alert event.
- **Cooldown Hysteresis:** Enforces a 5.0-second refractory timer to prevent webhook flooding.
- **Speaker Notes:** "Our calibrated hierarchy and 5-second cooldown eliminate alarm fatigue, while emitting full object counts and actor coordinates to the security team."

---

### Slide 14: Step 10 — Threat-5 Empirical Benchmark Results
- **Header:** Step 10: State-of-the-Art Benchmark Results
- **Primary Benchmark Performance (RLVD Test Split, $N = 300$):**
  - **ROC-AUC: 0.9818 (98.18%)** (+44.02% over baseline)
  - **Average Precision: 0.9819 (98.19%)**
  - **Accuracy: 92.67%** | **Precision: 90.00%** | **Recall: 96.00%** | **F1-Score: 0.9290**
  - **Throughput: 31.2 FPS on CPU** (>105 FPS on GPU)
- **Cross-Domain Zero-Shot Evaluation (UCF-Crime Test Split, $N = 290$):**
  - **Recall: 80.00%** (Catches 112 out of 140 crimes zero-shot) | **F1: 0.6038** | **33.55 FPS**
- **Test Suite Verification:**
  - **58 / 58 Tests Passing (100% Success)** across threat semantics, checkpoints, and inference pipelines.
- **Speaker Notes:** "On the real-world violence benchmark, ARGUS Threat-5 reaches 98.18% ROC-AUC and 92.9% F1-score at 31 frames per second on a standard CPU. When transferred zero-shot to UCF-Crime, it maintains 80% recall."

---

### Slide 15: Conclusion, Key Contributions & Q&A
- **Header:** Conclusion & Research Contributions
- **Summary of Core Contributions:**
  1. **5-Class Threat Graph Ontology:** Explicitly isolates Person, Vehicle, Bag, Weapon, and Other with 12D node and 10D relational threat edge conditioning.
  2. **Vectorized Pure-PyTorch EdgeGAT:** Sub-1.5 ms edge-conditioned message passing without external C++ dependencies.
  3. **Event-Driven Adaptive Sampling:** Slashes 48.7% of redundant static frames while bounding max gap ($G_{\max}=4$).
  4. **Graph-Guided Token Pruning:** Saliency-based pruning cutting transformer self-attention FLOPs by $4\times$.
  5. **Strictly Causal Streaming Memory:** $\mathcal{O}(1)$ bounded FIFO queue guaranteeing continuous 24/7 edge viability.
- **Academic Authors:** Omkar Joshi & Sameer Kumar | School of CSE, VIT Vellore
- **Speaker Notes:** "In conclusion, ARGUS Threat-5 proves that physical relational graph architectures outperform monolithic 3D CNNs and rival massive VLMs in accuracy while running in real time on edge CPUs. We are now open for questions."
