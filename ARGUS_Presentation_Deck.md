# ARGUS: Presentation Slide Deck (15 Slides)
## Adaptive Relational Graph Understanding for Surveillance
**Authors:** Omkar Joshi, Sameer Kumar  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore  
**Guide / Review:** Academic Project Review & Defense  

---

### Slide 1: Title & Overview
- **Header:** ARGUS: Adaptive Relational Graph Understanding for Surveillance
- **Sub-header:** An Edge-Viable Causal Graph-Transformer Framework for Real-Time Video Anomaly Detection
- **Presenter Information:** Omkar Joshi & Sameer Kumar | School of CSE, VIT Vellore
- **Key Badges:** 
  - 0.7949 ROC-AUC Joint Checkpoint
  - 13.6–16.0 FPS on Standard CPU (75+ FPS on Edge GPU)
  - Strict Online Temporal Causality ($G_{\max} = 4$)
- **Visual Suggestion:** Dual-pane split: on the left, ARGUS logo and institutional header; on the right, screenshot of the ARGUS Cyber-HUD surveillance player with actor bounding boxes and EdgeGAT relational lines.
- **Speaker Notes:** "Good morning, respected professors. Today we present ARGUS, an edge-deployable, causal graph-transformer neural framework for real-time video anomaly detection in surveillance networks."

---

### Slide 2: The Surveillance Paradox & Motivation
- **Header:** The Problem: Why Current Surveillance Systems Fail
- **Core Dilemma:** The *Surveillance Paradox* — billions of CCTV cameras exist worldwide, yet 99% of surveillance remains purely forensic (recorded passively, reviewed only after a crime occurs).
- **Three Core Operational Bottlenecks:**
  1. **Human Vigilance Fatigue:** Operators miss up to 45% of critical incidents after 12 minutes, and up to 95% after 22 minutes.
  2. **Compute & Bandwidth Saturation:** Over 95% of 24/7 CCTV video is empty, static background. Streaming raw uncompressed video across networks is unsustainable.
  3. **The Relational Nature of Physical Violence:** An altercation is not a static texture. Two people walking together is normal; two people converging at high velocity with striking motions is an assault. Pixel-grid models fail to distinguish these dynamics.
- **Speaker Notes:** "Human operators cannot watch dozens of monitors continuously. Meanwhile, traditional deep learning models waste 95% of their compute on empty walls and static pavements."

---

### Slide 3: The Failure of Existing Paradigms
- **Header:** Limitations of Prior Art: 3D CNNs vs. Large VLMs
- **Comparison Table:**
  | Dimension | 3D CNNs / VideoMAE | Large VLMs (7B+ Params) | ARGUS (Our Approach) |
  |---|---|---|---|
  | **Representation** | Monolithic 3D pixel grid | Autoregressive text tokens | **Dynamic Scene Graph** |
  | **Compute Footprint** | Heavy 3D convolutions | Prohibitive (>14 GB VRAM) | **Lightweight (Edge-Viable)** |
  | **Latency** | 150–300 ms per clip | 4–8 seconds (1.4 FPS) | **45 ms / 16 FPS on CPU** |
  | **Operational Risk** | Black-box opacity | Hallucination liability | **Deterministic & Explainable** |
  | **Edge Viability** | Requires server GPU | Impossible on edge CCTV | **Pure PyTorch CPU/NPU** |
- **The VLM Trap:** A physical assault lasts 2–3 seconds. A 7B parameter VLM running at 1.5 FPS takes 6 seconds to respond — by then, the incident is over.
- **Speaker Notes:** "While modern research has chased 7-billion parameter vision-language models, they are unusable on edge cameras due to latency and hallucinations. ARGUS treats anomalies not as language, but as physical topological disruptions."

---

### Slide 4: End-to-End System Architecture (Dual-Layer Topology)
- **Header:** System Topology: Decoupled Dual-Layer Engine
- **Outer Layer (Operational Infrastructure):**
  - Binary-Safe Multipart Stream Uploader
  - Pre-Trained & Fine-Tuned ConvNeXt Spatial Feature Backbone
  - Semantic Scene Graph & Bilinear RoI Extractor
  - Multi-Tier Hysteresis Alert Dispatcher & Live Web Console
- **Inner Layer (High-Throughput Core Neural Engine):**
  - Event-Driven Adaptive Temporal Sampler ($G_{\max} = 4$)
  - Vectorized EdgeGAT (Edge-Conditioned Graph Attention)
  - Topology-Aware Gated Cross-Attention Fusion
  - Graph-Guided Visual Token Pruner (50% token reduction $\rightarrow 4\times$ FLOP savings)
  - Causal Temporal Transformer with 16-Clip Streaming FIFO Memory
- **Speaker Notes:** "ARGUS decouples operational video handling from the neural reasoning engine, allowing sub-2-second inference on standard commodity hardware."

---

### Slide 5: Step 1 — Video Ingestion & Binary Stream Parsing
- **Header:** Step 1: Ingestion, Decoding & Byte-Range Setup
- **What Happens When a Video is Uploaded:**
  1. **Binary-Safe Multipart Stream Ingestion:** Multi-threaded `ThreadingHTTPServer` parses incoming binary payload without MIME corruption (handles MP4/AVI/MOV up to 100MB).
  2. **Metadata & Property Extraction:** OpenCV reads native framerate (e.g., 25/30 FPS), total frame count, aspect ratio, and video duration.
  3. **HTTP 206 Partial Content Preparation:** Enables byte-range streaming (`Range: bytes=start-end`), allowing zero-latency timeline seeking on the web client.
  4. **Standardized Spatial Scaling:** Decodes frames and normalizes tensor dimensions to $128 \times 128$ for optimal edge throughput.
- **Speaker Notes:** "The pipeline begins the instant a video file or live stream is received. We immediately extract physical video metadata and set up partial-content streaming so the user experiences zero buffering."

---

### Slide 6: Step 2 — Event-Driven Adaptive Temporal Sampling
- **Header:** Step 2: Adaptive Sampling (Filtering 50% Redundant Frames)
- **The Mathematical Formulation:**
  - Evaluates inter-frame kinetic changes using three lightweight indicators:
    $$\text{Pixel Difference: } D_t = \frac{1}{HW}\sum |I_t - I_{t-1}|$$
    $$\text{Spatial Gradient Energy: } G_t = \|\nabla I_t\|_1$$
    $$\text{Shannon Entropy Divergence: } H_t \text{ over 32-bin intensity histograms}$$
  - Composite motion metric: $M_t = \alpha D_t + \beta G_t + \gamma H_t \ge \tau$
- **Hard Coverage Bound ($G_{\max} = 4$):**
  - If 4 consecutive frames fall below threshold $\tau$, a frame is forcibly ingested.
  - *Guarantees creeping intrusions and slow-moving anomalies are never missed.*
- **Outcome:** **31% to 50% frame reduction** without sacrificing anomaly recall.
- **Speaker Notes:** "Instead of blindly processing all 30 frames every second, Module 1 uses motion gradients and entropy to skip empty frames, while the hard bound of 4 guarantees slow-moving events are never skipped."

---

### Slide 7: Step 3 — Spatial Backbone & Semantic Actor Detection
- **Header:** Step 3: ConvNeXt Feature Extraction & Entity Grounding
- **Spatial Feature Backbone:**
  - Jointly fine-tuned ConvNeXt-Tiny architecture.
  - Fine-tuned end-to-end with differential learning rates ($0.1\times$ backbone, $1.0\times$ head) to handle CCTV compression artifacts and motion blur.
- **Semantic Entity Detection:**
  - Detects active actors, vehicles, and critical objects.
  - **10D Kinematic & Semantic Node Vector:**
    $$v_i = [c_x, c_y, w, h, v_x, v_y, \text{class}_{\text{one-hot}}, \text{conf}]$$
  - **64D Bilinear RoI Visual Descriptor:** Extracts high-resolution visual appearance directly from ConvNeXt feature maps.
  - **Total Node Representation:** 74-dimensional feature vector per detected entity.
- **Speaker Notes:** "Here we isolate individual entities. Each actor receives a 74-dimensional descriptor encoding their spatial position, instantaneous velocity vector, bounding box size, and visual feature crops."

---

### Slide 8: Step 4 — Dynamic Relational Scene Graph Construction
- **Header:** Step 4: Modeling Physical Interactions (Scene Graph)
- **Constructing Graph Edges:**
  - Directed edges are instantiated between actor pairs within proximity threshold $d < 0.42$.
- **Continuous 7D Relational Edge Conditioning:**
  $$e_{ij} = \big[\text{dist}(i,j),\, \text{IoU}(i,j),\, \Delta v_{ij},\, \text{persistence},\, \mathbf{1}_{\text{P2P}},\, \mathbf{1}_{\text{P2O}},\, \mathbf{1}_{\text{O2O}}\big]$$
  - $\Delta v_{ij}$: Relative kinetic velocity (converging vs. diverging).
  - $\text{IoU}(i,j)$: Physical contact or spatial overlap.
- **Three Explicit Interaction Categories:**
  - **Person-to-Person (P2P):** Brawls, physical assaults, wrestling, pushes.
  - **Person-to-Object (P2O):** Weapon pulling, bag snatching, vandalism.
  - **Object-to-Object (O2O):** Vehicle crashes, barrier collisions.
- **Speaker Notes:** "This is the core differentiator of ARGUS. We explicitly model relationships. Two people walking together have low relative velocity; during an assault, relative velocity and IoU overlap spike dramatically."

---

### Slide 9: Step 5 — Vectorized EdgeGAT (Graph Attention Network)
- **Header:** Step 5: Relational Message Passing via EdgeGAT
- **Why Standard GNNs Fail:** Standard GCNs and GATs treat edges as binary 0 or 1 links, ignoring the continuous physical kinematics between actors.
- **Mathematical Attention Formulation:**
  $$\alpha_{ij} = \frac{\exp\Big(\text{LeakyReLU}\big(a^T [W_v h_i \,\|\, W_v h_j \,\|\, W_e e_{ij}]\big)\Big)}{\sum_{k \in \mathcal{N}_i} \exp\Big(\text{LeakyReLU}\big(a^T [W_v h_i \,\|\, W_v h_k \,\|\, W_e e_{ik}]\big)\Big)}$$
- **Node State Update:**
  $$h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W_v h_j\right)$$
- **High-Performance Vectorization:**
  - Built with pure PyTorch scatter operations. Zero Python loops.
  - Executes in **$<1.5\text{ ms}$ per frame** on standard CPU.
- **Speaker Notes:** "EdgeGAT modulates attention by physical edge features. If two people are colliding at high speed, the attention weight $\alpha_{ij}$ spikes, focusing the network on that specific interaction."

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
- **Impact:** **50% sequence length reduction**, achieving **$4\times$ FLOP savings** in the subsequent temporal transformer.
- **Speaker Notes:** "By pruning background tokens that contain no human interaction, we reduce the sequence length by half and cut transformer computational complexity by a factor of 4."

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
- **Speaker Notes:** "Our loss function ensures three properties: high classification accuracy, smooth non-flickering predictions across frames, and temporal sparsity so baseline scores stay close to zero."

---

### Slide 13: Step 9 — Operational Alert Dispatcher & Hysteresis
- **Header:** Step 9: Alert Dispatching & Alarm Fatigue Prevention
- **Eliminating False Alarms:** Normal pedestrian walking (`NV_344`, `NV_950`) scores under $0.05$ ($<5\%$).
- **Calibrated Multi-Tier Severity Hierarchy:**
  | Tier | Score Range | Operational Meaning | Action Triggered |
  |---|:---:|---|---|
  | **NORMAL** | $< 0.35$ | Routine pedestrian flow | Silent logging |
  | **ELEVATED** | $0.35 - 0.50$ | Suspicious clustering / fast approach | Flag for review |
  | **HIGH** | $0.50 - 0.75$ | Confrontation / robbery | Audio alert chime |
  | **CRITICAL** | $\ge 0.75$ | Verified assault / physical violence | Security dispatch |
- **Cooldown Hysteresis:** Enforces a 5.0-second refractory cooldown timer to prevent alert flooding during prolonged altercations.
- **Speaker Notes:** "A surveillance system that cries wolf is turned off by security guards. Our calibrated 0.35 threshold and 5-second cooldown eliminate alarm fatigue while ensuring 100% recall on genuine assaults."

---

### Slide 14: Step 10 — The Live Web Console & Verification
- **Header:** Step 10: Interactive UI & Benchmark Results
- **Features of the Production Web Platform (`web_demo.py`):**
  - **Dual-Mode Player:** Native 30 FPS video playback with synchronized cyber-HUD + AI Vision keyframe inspector with actor bounding boxes and EdgeGAT lines.
  - **Synchronized Anomaly Timeline:** Interactive Chart.js timeline with click-to-seek scrubbing.
  - **Real-Time Sensitivity Slider:** Allows dynamic adjustment of threshold (0.10 to 0.85) with instant client-side recalculation.
- **Empirical Benchmark Performance (605 Test Clips across UCF-Crime & RealLifeViolence):**
  - **ROC-AUC: 0.7949** (+25.33% over baseline)
  - **Average Precision: 0.7449**
  - **Accuracy: 72.73%** | **F1-Score: 0.7038**
  - **Speed:** 13.6–16.0 FPS (CPU) / 75+ FPS (GPU) | **Tests:** 27/27 Passing (100%)
- **Speaker Notes:** "The final step presents the telemetry to the operator. The web interface streams at full 30 FPS, highlights the interacting actors, and displays the exact causal anomaly score curve in real time."

---

### Slide 15: Conclusion, Key Contributions & Q&A
- **Header:** Conclusion & Research Contributions
- **Summary of Core Contributions:**
  1. **Dual-Layer Architecture:** Decouples operational video streaming from core neural graph reasoning.
  2. **Event-Driven Sampling with Bound:** Adaptive sampling with $G_{\max} = 4$ saves up to 50% frames while preserving slow anomalies.
  3. **Vectorized EdgeGAT:** First pure-PyTorch edge-conditioned graph attention for surveillance modeling P2P, P2O, and O2O relations.
  4. **Graph-Guided Token Pruning:** Saliency-based token pruning achieving a $4\times$ FLOP reduction.
  5. **Strictly Causal Streaming:** $\mathcal{O}(1)$ constant memory FIFO transformer enabling true 24/7 continuous deployment.
- **Academic Authors:** Omkar Joshi & Sameer Kumar | School of CSE, VIT Vellore
- **Speaker Notes:** "In conclusion, ARGUS demonstrates that physical scene graphs combined with causal transformers outperform massive black-box models in both speed, accuracy, and interpretability. We are now open for questions."
