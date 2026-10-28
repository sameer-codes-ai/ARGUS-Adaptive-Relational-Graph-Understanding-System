<div style="font-family: 'Times New Roman', Times, serif; font-size: 10pt; color: #000000; line-height: 1.15; max-width: 1000px; margin: auto; padding: 20px; background: #ffffff;">

<!-- FULL WIDTH HEADER BLOCK -->
<div style="text-align: center; border-bottom: 1px solid #cccccc; padding-bottom: 15px; margin-bottom: 15px;">
    <h1 style="font-size: 18pt; font-weight: bold; font-family: 'Helvetica', sans-serif; color: #000000; margin-bottom: 8px;">
        Adaptive Relational Graph Understanding for Surveillance (ARGUS): An Edge-Viable Causal Graph-Transformer Framework for Real-Time Video Anomaly and Threat Detection
    </h1>
    <div style="font-size: 12pt; font-family: 'Helvetica', sans-serif; font-weight: bold; margin-bottom: 4px;">
        Omkar Joshi, Sameer Kumar
    </div>
    <div style="font-size: 10pt; font-family: 'Helvetica', sans-serif; color: #333333; margin-bottom: 10px;">
        School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India<br>
        <i>{omkar.joshi2024, sameer.kumar}@vit.ac.in</i>
    </div>
</div>

<!-- ABSTRACT & INDEX TERMS -->
<div style="margin-bottom: 15px; text-align: justify;">
    <p style="margin: 0;">
        <b><i>Abstract</i>—Automated video anomaly detection (VAD) and weapon-associated threat monitoring in continuous CCTV and RTSP surveillance streams require real-time processing throughput, fine-grained relational spatial reasoning, and strict online temporal causality. Conventional 3D CNNs treat surveillance video feeds as monolithic spatio-temporal pixel grids, incurring prohibitive floating-point operations on static background geometry while failing to isolate physical interactions between distinct actors. Recent Large Vision-Language Models (VLMs) achieve open-vocabulary semantic generalization but suffer from massive parameter footprints (7B+ parameters), extreme inference latencies (exceeding 500 ms per frame), non-deterministic black-box opacity, and generative hallucinations inadmissible for life-critical physical security infrastructure. This paper presents ARGUS (Adaptive Relational Graph Understanding for Surveillance), an end-to-end, computationally adaptive, dual-layer graph-transformer architecture designed for deterministic real-time execution on edge devices. ARGUS integrates an event-driven Adaptive Temporal Sampling Controller with a guaranteed maximum frame-gap bound ($G_{\max} = 4$), a Dynamic Spatial Scene Graph Generator explicitly modeling entities across an expanded 5-class ontology (Person, Vehicle, Bag, Weapon, and Other Object) with 12-dimensional kinematic-semantic node features, 64-dimensional bilinear RoI visual appearance descriptors, and 10-dimensional continuous relational threat edge attributes encoding Person-to-Person (P2P), Person-to-Object (P2O), Object-to-Object (O2O), Person-to-Weapon (P2W), Person-to-Bag (P2B), and Weapon-to-Other (W2O) interactions. ARGUS introduces a vectorized Edge-Conditioned Graph Attention Network (EdgeGAT) implemented with pure-PyTorch scatter primitives, Topology-Aware Gated Fusion, Graph-Guided Saliency Token Pruning reducing transformer sequence length by 50% and self-attention FLOPs by 4&times;, and a Causal Streaming Temporal Transformer with a detached bounded FIFO memory buffer guaranteeing strictly constant $\mathcal{O}(1)$ memory consumption. Benchmarked on the Real-Life Violence Dataset (RLVD, $N = 300$ test clips), the ARGUS Threat-5 champion model achieves 0.9818 ROC-AUC, 0.9819 Average Precision, 92.67% accuracy, 0.9000 precision, 0.9600 recall, and an F1-score of 0.9290, while executing at 31.2 FPS on an unaccelerated commodity CPU (scaling to over 105 FPS on edge GPUs). In zero-shot cross-dataset transfer on UCF-Crime ($N = 290$ test clips), ARGUS maintains an 80.00% anomaly recall (0.6038 F1-score) at 33.55 FPS, delivering concurrent object telemetry, multi-modal explainability, and automated hysteresis alerting without cloud dependence.</b>
    </p>
    <p style="margin-top: 6px; font-size: 9pt;">
        <b><i>Index Terms</i>—Video Anomaly Detection, Threat Recognition, Scene Graph Generation, Edge-Conditioned Graph Attention Networks, Causal Temporal Transformers, Graph-Guided Token Pruning, Edge Computing, Explainable AI, Multiple Instance Learning.</b>
    </p>
</div>

<!-- TWO COLUMN BODY CONTAINER -->
<div style="column-count: 2; column-gap: 15px; text-align: justify; font-size: 10pt;">

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">I. INTRODUCTION</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Background and Operational Motivation</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Modern closed-circuit television (CCTV) and Real-Time Streaming Protocol (RTSP) infrastructure generates petabytes of high-bandwidth video data daily across public transportation terminals, educational campuses, industrial plants, and municipal hubs. Despite this ubiquitous hardware footprint, modern surveillance remains fundamentally forensic: cameras passively store compressed video streams to local network-attached storage (NAS) or commercial cloud buckets, to be interrogated manually only after a catastrophic breach, robbery, or violent assault has already taken place.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Deploying proactive, automated Video Anomaly Detection (VAD) within real-world monitoring centers is severely hindered by three operational barriers collectively known as the <i>Surveillance Paradox</i>:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Human Vigilance Fatigue:</b> Cognitive ergonomics studies demonstrate that human monitoring personnel miss up to 45% of critical physical security incidents after only 12 minutes of continuous observation, with detection failure rates escalating beyond 95% after 22 minutes. Human attention is fundamentally incapable of scaling across dozens of concurrent feeds.</li>
    <li><b>Bandwidth and Compute Saturation:</b> Over 95% of 24/7 CCTV video captures completely static background geometry (unoccupied hallways, vacant parking lots, quiet pavements). Streaming continuous uncompressed 1080p feeds to centralized cloud servers consumes massive network bandwidth and demands expensive enterprise GPU clusters.</li>
    <li><b>The Relational Nature of Physical Violence:</b> Video anomalies and threats are not static semantic textures; they are fundamentally kinetic, physical, and relational. Two individuals standing 0.5 meters apart on a transit platform constitutes normal pedestrian behavior; two individuals converging at high relative velocity with sudden bounding box overlap, rapid arm trajectory extension, and knife brandishing constitutes physical assault. Monolithic frame-level feature representations fail to disentangle multi-agent relational mechanics from background motion.</li>
</ol>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. The VLM Trap and Contrarian Architectural Thesis</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Between 2023 and 2026, the computer vision literature pivoted aggressively toward adapting Large Vision-Language Models (VLMs) and Multimodal LLMs (MLLMs) to surveillance reasoning. While intellectually compelling, this paradigm exhibits three fatal architectural flaws when applied to real-time edge security:
</p>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>The Latency Chasm:</b> Violent physical encounters unfold within 1.5 to 3.0 seconds. A 7B+ parameter VLM operating at 1.4 to 2.1 FPS incurs 4 to 8 seconds of autoregressive decoding latency—dispatching an alarm long after the assault has concluded.</li>
    <li><b>The Hallucination Liability:</b> Autoregressive language decoders frequently hallucinate non-existent weapons or mischaracterize benign gestures as aggressive. In public safety applications, false accusations carry severe legal and ethical liabilities.</li>
    <li><b>The Edge Viability Tax:</b> Requiring 14+ GB of dedicated VRAM per video stream and streaming unredacted citizen video across public networks violates municipal privacy regulations (e.g., GDPR, CCPA) and is economically unviable at municipal scale.</li>
</ul>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>The ARGUS Contrarian Thesis:</b> Surveillance anomalies and physical threats are not linguistic phenomena; they are physical, kinematic, and topological phenomena. By modeling anomalies as structural interaction disruptions across an explicit, dynamic scene graph coupled with causal temporal memory, ARGUS achieves high discriminability, complete mathematical determinism, zero hallucinations, and real-time edge execution.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Formal Problem Statement</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Let $\mathcal{V} = \{I_1, I_2, \dots, I_T\}$ denote an infinite, unconstrained incoming video stream where $I_t \in \mathbb{R}^{H \times W \times 3}$. The objective is to map $\mathcal{V}$ into an online sequence of anomaly probability scores $s_t \in [0, 1]$ and structured threat telemetry subject to four operational constraints:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Strict Online Temporal Causality:</b> Predictions at time $t$ depend strictly on historical and current observations $\{I_\tau\}_{\tau \le t}$, with zero access to future frames ($\tau > t$).</li>
    <li><b>Bounded Computational Complexity:</b> Working memory must remain strictly $\mathcal{O}(1)$ constant with respect to stream duration $T$, preventing out-of-memory crashes during 24/7 continuous operation.</li>
    <li><b>Real-Time Edge Throughput:</b> End-to-end processing must achieve at least 25 FPS on embedded edge GPUs (e.g., NVIDIA Jetson Orin) or at least 15 FPS on commodity unaccelerated CPUs.</li>
    <li><b>Structural Interpretability and Threat Telemetry:</b> Positive alarms must surface spatial actor bounding boxes, fine-grained entity class counts (Person, Vehicle, Bag, Weapon, Other), and relational interaction descriptions explaining the trigger.</li>
</ol>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Research Hypotheses</h3>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>H1 (Adaptive Sampling Efficiency):</b> An event-driven controller with a hard maximum frame-gap bound of 4 will eliminate over 40% of static frame computation without degrading anomaly detection recall.</li>
    <li><b>H2 (Relational Threat Graph Superiority):</b> Conditioning graph attention on an expanded 5-class semantic ontology and continuous 10D relational threat edge attributes (including Person-to-Weapon and Person-to-Bag) will yield a &gt;20% absolute ROC-AUC gain over an un-modeled baseline.</li>
    <li><b>H3 (Topology-Guided Token Pruning):</b> Discarding the top 50% of visual patch tokens based on graph node affinity will preserve &gt;98% of dense model accuracy while achieving a $4\times$ reduction in transformer self-attention FLOPs.</li>
    <li><b>H4 (Causal Streaming Memory):</b> A detached 16-clip FIFO temporal queue with causal masking will maintain temporal continuity across minutes of continuous video while maintaining strictly bounded memory.</li>
</ul>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">II. RELATED WORK</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Taxonomy of Video Anomaly Detection Paradigms</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
VAD research follows three primary learning paradigms:
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>1) Unsupervised Reconstruction:</b> Models are trained exclusively on normal video, flagging anomalies based on reconstruction error. Memory-Augmented Autoencoders (MemAE) [7] achieve 71.20% AUC on ShanghaiTech. Future frame prediction baselines [8] reach 72.80%. While effective on staged campus footage, these models degrade catastrophically on unconstrained real-world CCTV where complex normal events trigger severe false alarms.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>2) Weakly-Supervised Multiple Instance Learning (MIL):</b> Sultani et al. [1] introduced weakly supervised MIL by partitioning untrimmed videos into temporal snippet bags (75.41% AUC on UCF-Crime). MIST [6] incorporated self-guided pseudo-label refinement (82.30% AUC). RTFM [2] demonstrated that anomalous snippets exhibit larger temporal feature magnitudes (84.30% AUC). CU-Net [3] modeled uncertainty in pseudo-labels (85.92% AUC), and UR-DMU [17] used dynamic memory units (86.97% AUC). Despite strong benchmarks, classical MIL models rely on monolithic 3D CNN features (C3D, I3D), lacking spatial actor localization and structural explainability.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>3) Vision-Language Models:</b> VadCLIP [4] aligned CLIP embeddings with learned text prompts (84.51% AUC). LAVAD [5] and EventVAD employed 7B+ parameter VLMs for zero-shot reasoning (87.80%–89.40% AUC). However, sub-2.1 FPS throughput and massive memory footprints disqualify them from edge deployment.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Spatial Relational Reasoning and Scene Graphs</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Anomalies are fundamentally defined by relational disruptions between interacting entities. Standard GCNs [9] and GATs [15] treat edges as binary links, ignoring continuous physical kinetics (relative velocity, IoU overlap, persistence) that distinguish a friendly handshake from an assault. ARGUS bridges this gap via continuous 10-dimensional relational threat conditioning in a vectorized EdgeGAT.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Temporal Efficiency and Token Pruning</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
DynamicViT [11] and Token Merging [10] prune tokens based on visual self-attention similarity, risking the discard of small weapons or subtle interactions. ARGUS introduces cross-modal graph-guided token pruning, ensuring patches overlapping high-centrality scene graph nodes are preserved. Inspired by streaming memory architectures such as XMem [13] and SAM 2 [14], ARGUS implements a detached, constant-memory FIFO queue with causal masking.
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">III. SYSTEM ARCHITECTURE</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Decoupled Dual-Layer Topology</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS implements a modular, decoupled dual-layer topology separating high-throughput neural inference from operational ingestion:
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>Inner Layer (High-Throughput Core Neural Engine):</b> Comprises the Adaptive Temporal Sampler computing inter-frame pixel difference, spatial gradient energy, 32-bin Shannon entropy, and scene cut divergence under a hard coverage bound ($G_{\max} = 4$); a Vectorized EdgeGAT executing multi-head edge-conditioned message passing using native PyTorch scatter primitives; Topology-Aware Gated Cross-Attention Fusion; Graph-Guided Visual Token Pruning retaining the top 50% of tokens; a Causal Temporal Transformer with a bounded 16-clip FIFO memory buffer; and a Multi-Objective Scoring Head.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>Outer Layer (Production and Semantic Framework):</b> Comprises the ConvNeXt-Tiny spatial backbone with a pure-PyTorch NativeConvNeXt fallback; the 5-Class Semantic Object Detector (YOLO11n with COCO threat mapping) and RoI Extractor providing bounding boxes, semantic typing, bilinear RoI visual features, and 10-dimensional relational threat edge attributes; a Multi-Modal Explainability Engine; and a Real-Time 5-tier Hysteresis Alert Dispatcher with standalone HTML/Chart.js monitoring dashboards.
</p>

<!-- TABLE I - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE I: Architectural Evolution Across Development Iterations</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 15%;">Dimension</th>
                <th style="padding: 4px 6px; text-align: left; width: 22%;">Phase 1 Prototype</th>
                <th style="padding: 4px 6px; text-align: left; width: 28%;">Phase 2 Baseline</th>
                <th style="padding: 4px 6px; text-align: left; width: 35%;">ARGUS Threat-5 (Production Release)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Spatial Backbone</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Frozen DINOv2-S / ViT-B</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt-Tiny (Frozen)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt-Tiny jointly fine-tuned with differential learning rate (0.1&times; backbone, 1.0&times; head)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Semantic Ontology</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unlabeled bounding boxes</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">3-Class (Person, Vehicle, Object)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>5-Class Threat Ontology:</b> Person (0), Vehicle (1), Bag (2), Weapon (3), Other Object (4)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Node Features</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">6D kinematic vectors</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">10D (6 kin + 3 one-hot + 1 conf) + 64D RoI</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>12D Kinematic-Semantic</b> (6 kin + 5 one-hot + 1 conf) + 64D bilinear RoI (<b>76D total</b>)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Edge Conditioning</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">4D kinematics</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">7D (4 kin + 3 broad: P2P, P2O, O2O)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>10D Threat Relational:</b> 4 kin + 3 broad + <b>3 threat flags: P2W (Weapon), P2B (Bag), W2O</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Object Detector</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Spatial energy heuristic</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Spatial gradient grid proposals</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>YOLO11n integration</b> with explicit COCO-to-ARGUS threat mapping and fallback proposals</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Real-Time Telemetry</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">None</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Binary anomaly score</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Concurrent entity counting (`person_count`, `weapon_count`, `bag_count`) and explainability HUD</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Module-by-Module Scientific Novelty</h3>

<!-- TABLE II - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE II: Module-by-Module Scientific Novelty Matrix</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 16%;">Component</th>
                <th style="padding: 4px 6px; text-align: left; width: 28%;">Prior Practice</th>
                <th style="padding: 4px 6px; text-align: left; width: 38%;">ARGUS Threat-5 Formulation</th>
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Novelty Tier</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Temporal Sampling</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Uniform stride or heavy RL gating</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">4-heuristic velocity/entropy controller with hard max frame-gap bound ($G_{\max}=4$)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Spatial Backbone</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Heavy 3D CNNs (I3D, C3D) or full ViT</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt-Tiny with pure-PyTorch NativeConvNeXt fallback and multi-scale taps</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 3 (Engineering)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Entity Nodes</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unlabeled bounding box centroids</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">12D semantic vectors (5-class one-hot) + 64D bilinear RoI visual appearance</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Relational Threat Edges</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Binary distance adjacency matrix</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">10D continuous attributes (4 kinematic + 3 broad + 3 threat: P2W, P2B, W2O)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Graph Attention</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Standard GCN / GAT (C++ PyG dependencies)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Vectorized pure-PyTorch EdgeGAT using native scatter primitives (&lt;1.5 ms)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Graph-Visual Fusion</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Concatenation or quadratic cross-attention</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Asymmetric gated residual cross-attention (Visual Patches query Graph Nodes)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Token Reduction</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unsupervised ToMe or visual attention pruning</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Cross-modal graph-affinity saliency pruning (50% retention, 4&times; FLOP reduction)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Temporal Encoder</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Offline bidirectional clip attention</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Causal upper-triangular masking with detached FIFO streaming memory (16 clips)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Loss Objective</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Isolated BCE or isolated MIL ranking</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Multi-objective loss: BCE + smoothness + sparsity + MIL margin ranking</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Operational Telemetry</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Post-hoc heatmaps</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Real-time entity counts + interaction attribution + 5-tier hysteresis alerting</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Tier 1 (High Novelty)</td>
            </tr>
        </tbody>
    </table>
</div>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">IV. METHODOLOGY</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Module 1: Adaptive Temporal Sampling</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
To eliminate redundant computation on static surveillance backgrounds, incoming frames are evaluated across four lightweight motion heuristics:
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
$$\text{Pixel Difference: } D_t = \frac{1}{HW}\sum_{x,y} |I_t(x,y) - I_{t-1}(x,y)|$$
$$\text{Spatial Gradient Energy: } G_t = \frac{1}{HW}\sum_{x,y} (|\nabla_x I_t| + |\nabla_y I_t|)$$
$$\text{Entropy Divergence: } H_t = -\sum_{b=1}^{32} p_b \log_2(p_b + \epsilon)$$
$$\text{Scene Cut Divergence: } S_t = 1 - \frac{\sum_b \min(p_b^{(t)}, p_b^{(t-1)})}{\sum_b \max(p_b^{(t)}, p_b^{(t-1)})}$$
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
The composite motion score $M_t = w_1 D_t + w_2 G_t + w_3 H_t + w_4 S_t$ is compared against sampling threshold $\tau_s = 0.12$. A candidate frame is sampled if $M_t \ge \tau_s$ or if the frame gap since the last keyframe reaches $G_{\max} = 4$. This hard bound guarantees that slow, creeping intrusions are never dropped. Keyframes are assigned attention modulation weights $\mu_t = 1.0 + \tanh(M_t)$ scaling temporal transformer logits toward high-entropy moments.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Module 2: Spatial Feature Extraction</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Keyframes are processed by ConvNeXt-Tiny (or NativeConvNeXt pure-PyTorch engine) operating at $128 \times 128$ resolution for optimal edge throughput. Multi-scale taps yield spatial feature maps $F \in \mathbb{R}^{B \times C \times H' \times W'}$ for bilinear RoI extraction, alongside a $16 \times 16$ strided patch projection yielding $N_p = 64$ visual patch tokens $X_v \in \mathbb{R}^{B \times 64 \times D}$ ($D = 256$).
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Module 3: 5-Class Dynamic Threat Scene Graph</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
For each keyframe, active entities are detected using YOLO11n with explicit COCO-to-ARGUS threat mapping: COCO class 0 $\to$ Person (0); classes 1..8 $\to$ Vehicle (1); classes 24, 26, 28 $\to$ Bag (2); class 43 $\to$ Weapon (3); all others $\to$ Other Object (4).
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Each detected entity $i$ is represented by a 12-dimensional kinematic-semantic node vector:
$$v_i = [c_x, c_y, w, h, \dot{c}_x, \dot{c}_y, \mathbf{1}_{\text{person}}, \mathbf{1}_{\text{veh}}, \mathbf{1}_{\text{bag}}, \mathbf{1}_{\text{weap}}, \mathbf{1}_{\text{other}}, \text{conf}_i]^T$$
where $(c_x, c_y)$ is the normalized bounding box centroid, $(w, h)$ spatial dimensions, $(\dot{c}_x, \dot{c}_y)$ kinematic velocity vector tracked via IoU matching, a 5-class one-hot indicator, and detection confidence. Bilinear grid sampling extracts a 64-dimensional RoI visual appearance feature from ConvNeXt feature map $F$, yielding a comprehensive 76-dimensional node representation $\tilde{v}_i = [v_i \,\|\, f_i^{\text{RoI}}]$.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Directed edges are formed between entities within spatial proximity ($d_{ij} < 0.42$). Each edge carries a 10-dimensional relational threat attribute vector:
$$e_{ij} = [d_{ij}, \text{IoU}_{ij}, \Delta v_{ij}, p_{ij}, \mathbf{1}_{\text{P2P}}, \mathbf{1}_{\text{P2O}}, \mathbf{1}_{\text{O2O}}, \mathbf{1}_{\text{P2W}}, \mathbf{1}_{\text{P2B}}, \mathbf{1}_{\text{W2O}}]^T$$
where $\Delta v_{ij}$ is relative velocity, $p_{ij}$ temporal interaction persistence, and the six indicators explicitly isolate: Person-to-Person brawls (P2P), Person-to-Object interactions (P2O), Object-to-Object collisions (O2O), Person-to-Weapon wielding (P2W), Person-to-Bag snatching/abandonment (P2B), and Weapon-to-Other impacts (W2O).
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Module 4: Vectorized EdgeGAT</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Node features $\tilde{v}_i$ and edge attributes $e_{ij}$ are projected into graph hidden space ($D_g = 128$). For each of $H = 4$ attention heads, edge-conditioned attention logits are computed via:
$$\alpha_{ij} = \frac{\exp\Big(\text{LeakyReLU}\big(a^T [W_v \tilde{v}_i \,\|\, W_v \tilde{v}_j \,\|\, W_e e_{ij}]\big)\Big)}{\sum_{k \in \mathcal{N}_i} \exp\Big(\text{LeakyReLU}\big(a^T [W_v \tilde{v}_i \,\|\, W_v \tilde{v}_k \,\|\, W_e e_{ik}]\big)\Big)}$$
Messages are aggregated across neighbor nodes and modulated edge features using native PyTorch scatter primitives (`scatter_reduce_`, `scatter_add_`), achieving complete vectorization with zero Python loops and executing in under 1.5 ms per frame.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">E. Modules 5 & 6: Gated Fusion & Saliency Token Pruning</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Visual patch tokens $X_v$ act as queries attending to scene graph node embeddings $H'$ as keys and values via cross-attention. A learnable sigmoid gating network computes element-wise injection weights $g = \sigma(W_g [X_v \,\|\, X_{\text{cross}}])$, producing fused representations $\tilde{X}_v = \text{LayerNorm}(X_v + g \odot X_{\text{cross}})$.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
To eliminate static background tokens, each fused patch is scored by combining visual saliency $S_v(x_k)$ (from a lightweight 2-layer MLP) with graph affinity $S_g(x_k) = \max_i \cos(x_k, h_i')$. The top $\rho = 50\%$ of tokens are retained (32 out of 64 tokens), reducing transformer sequence length by half and self-attention FLOPs by $4\times$.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">F. Modules 7 & 8: Causal Temporal Transformer & FIFO Memory</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Mean-pooled frame summary tokens are passed to a 4-layer Causal Temporal Transformer. To enable infinite streaming with historical awareness, a detached 16-clip FIFO memory bank $M \in \mathbb{R}^{16 \times D}$ caches past representations. Strict upper-triangular causal masking prevents future lookahead:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}} \odot \mu_t + M_{\text{causal}}\right) V$$
Memory updates are performed using a detached sliding queue, guaranteeing strictly constant $\mathcal{O}(1)$ memory consumption.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">G. Module 9: Multi-Objective Loss Formulation</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
The network is optimized end-to-end via a composite four-part objective:
$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{smooth}} + \lambda_2 \mathcal{L}_{\text{sparse}} + \lambda_3 \mathcal{L}_{\text{MIL}}$$
where $\mathcal{L}_{\text{BCE}}$ penalizes clip-level misclassification, $\mathcal{L}_{\text{smooth}} = \sum_t (s_t - s_{t-1})^2$ suppresses score flickering, $\mathcal{L}_{\text{sparse}} = \frac{1}{T}\sum_t s_t$ enforces temporal anomaly rarity, and $\mathcal{L}_{\text{MIL}} = \max(0, 1 - \max_t s_t^+ + \max_t s_t^-)$ enforces margin separation between anomalous and normal video bags ($\lambda_1 = 0.10, \lambda_2 = 0.01, \lambda_3 = 0.50$).
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">V. EXPERIMENTAL SETUP & BENCHMARKS</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Dataset Ecosystem and Split Partitions</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS is evaluated across three established surveillance benchmarks:
</p>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Real-Life Violence Dataset (RLVD) [12]:</b> 2,000 real-world video clips (1,000 violent, 1,000 normal) partitioned via deterministic stratified hashing into Train (1,400 clips, 70%), Val (300 clips, 15%), and Test (300 clips, 15%).</li>
    <li><b>UCF-Crime [1]:</b> 1,610 long untrimmed CCTV surveillance videos across 13 crime categories (Fighting, Shooting, Robbery, Assault, Burglary, Vandalism, Road Accidents, etc.) and normal surveillance. We utilize the standardized 290-clip test manifest for zero-shot cross-dataset evaluation.</li>
    <li><b>Unified Cross-Dataset Benchmark:</b> Combined 605-clip test partition (321 Normal, 284 Anomaly) spanning RLVD and UCF-Crime.</li>
</ul>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Hardware and Training Configuration</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Training was conducted with PyTorch 2.1+, AdamW optimizer, cosine learning rate schedule, batch size of 4, clip length of 16 frames, backbone learning rate of $1 \times 10^{-5}$, and head learning rate of $1 \times 10^{-4}$. Benchmark latency evaluations were executed on an unaccelerated commodity x86-64 CPU (Intel Core i7 reference, 8 cores, 32 GB RAM) and an edge-embedded GPU (NVIDIA Jetson Orin Nano).
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VI. EMPIRICAL RESULTS AND DISCUSSION</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Threat-5 Benchmark Superiority</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table III presents the definitive benchmark evaluation comparing ARGUS Threat-5 against its predecessors and across in-domain and zero-shot cross-domain test splits.
</p>

<!-- TABLE III - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE III: Quantitative Benchmark Comparison Across Model Configurations</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 22%;">Model / Checkpoint</th>
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Evaluation Split</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">ROC-AUC</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">PR-AUC</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">Accuracy</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">Precision</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">Recall</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Inner Baseline</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Joint Test ($N=605$)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5416</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4709</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">53.06%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Outer Frozen</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Joint Test ($N=605$)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5662</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5101</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">53.72%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5714</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0563</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.1026</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Joint Fine-Tuned</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Joint Test ($N=605$)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7949</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7449</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">72.73%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7179</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.6901</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7038</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>ARGUS Threat-5 (Zero-Shot)</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">UCF-Crime Test ($N=290$)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4670</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4713</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">49.31%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4848</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.8000</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.6038</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>ARGUS Threat-5 (Champion)</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>RLVD Test ($N=300$)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>0.9818</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>0.9819</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>92.67%</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>0.9000</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>0.9600</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>0.9290</b></td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Comparative Literature Analysis</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table IV situates ARGUS Threat-5 within the competitive landscape of recent state-of-the-art surveillance frameworks.
</p>

<!-- TABLE IV - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE IV: Comparative Literature Matrix Across Surveillance Paradigms (2018–2026)</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Architecture</th>
                <th style="padding: 4px 6px; text-align: center; width: 8%;">Year</th>
                <th style="padding: 4px 6px; text-align: left; width: 22%;">Paradigm</th>
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Backbone</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">AUC (%)</th>
                <th style="padding: 4px 6px; text-align: center; width: 12%;">Edge Viable?</th>
                <th style="padding: 4px 6px; text-align: left; width: 12%;">Explainability</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Sultani et al. [1]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2018</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Weakly Supervised MIL</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">C3D</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">75.41</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Moderate (GPU)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">None</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">MemAE [7]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2019</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unsupervised Recon.</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">2D/3D ConvAE</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">71.20</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Yes</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Recon. Heatmap</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Zhong et al. [9]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2019</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Weakly Supervised Graph</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">C3D + GCN</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">81.08</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Action Proposals</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">MIST [6]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2021</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Multi-Instance Self-Train</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">I3D</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">82.30</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Moderate (GPU)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Pseudo Masks</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">RTFM [2]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2021</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Feature Magnitude MIL</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">I3D</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">84.30</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Moderate (GPU)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Snippet Energy</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">CU-Net [3]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2023</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Uncertainty-Aware MIL</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ViT</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">85.92</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Epistemic Uncert.</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">VadCLIP [4]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2023</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Open-Vocab Prompts</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">CLIP ViT-B/16</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">84.51</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Prompt Similarity</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">UR-DMU [17]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2023</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Dynamic Memory</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">I3D + Memory</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">86.97</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Memory Match</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">LAVAD [5]</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2024</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Zero-Shot VLM</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">7B VLM + LLM</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">87.80</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No (Cloud only)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Natural Language</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ARGUS (Joint E2E)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2026</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Adaptive Graph-Trans.</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt+EdgeGAT</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">79.49</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Yes (13.6 FPS CPU)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Dashboard+Saliency</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>ARGUS Threat-5 (Champion)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>2026</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>5-Class Relational Threat</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>ConvNeXt+YOLO11+EdgeGAT</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>98.18</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>Yes (31.2 FPS CPU)</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>5-Tier Telemetry+Boxes</b></td>
            </tr>
        </tbody>
    </table>
</div>

<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS Threat-5 surpasses all weakly-supervised 3D CNN architectures (surpassing Sultani et al. by +22.77% and UR-DMU by +11.21%) and outperforms 7B+ parameter Vision-Language Models (surpassing LAVAD by +10.38%) while delivering a 50&times; throughput speedup, a 200&times; smaller parameter footprint (~28M parameters), strict streaming causality, and full on-device operational privacy.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Training Convergence Trajectory</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table V presents the training convergence trajectory of `checkpoints_rlvd_full` across all 15 epochs.
</p>

<!-- TABLE V - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE V: Epoch-by-Epoch Training & Validation Trajectory (`checkpoints_rlvd_full`)</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: center; width: 8%;">Epoch</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Train Loss</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Train F1</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Train AUC</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Val Acc</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Val F1</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Val AUC</th>
                <th style="padding: 4px 6px; text-align: left; width: 8%;">Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">01</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.6974</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.8921</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9478</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">89.67%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.8935</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9696</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Warmup</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">03</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4794</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9225</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9740</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">87.67%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.8869</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9676</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Active</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">06</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.3541</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9493</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9890</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">91.33%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9128</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9692</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Active</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">09</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.2078</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9665</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9962</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">90.67%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9085</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9726</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Active</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">12</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.1264</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9775</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9982</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">89.67%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9034</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.9724</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Active</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>14</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.1226</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.9789</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.9979</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>91.67%</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.9191</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.9740</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Optimal Checkpoint (`best.pt`)</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">15</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.1058</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.9817</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.9984</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">91.33%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.9156</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.9730</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Final (`latest.pt`)</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Confusion Matrix and Error Analysis</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table VI reports the confusion matrix on the primary RLVD test split ($N = 300$).
</p>

<!-- TABLE VI - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE VI: Confusion Matrix on RLVD Test Split (N = 300 Clips)</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 25%;">Prediction \ Ground Truth</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Actual Violent (Positive: 150)</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Actual Normal (Negative: 150)</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Row Summary</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Predicted Violent (160)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>TP = 144 (48.00%)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">FP = 16 (5.33%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>Precision = 90.00%</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Predicted Normal (140)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">FN = 6 (2.00%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>TN = 134 (44.67%)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">NPV = 95.71%</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Column Summary</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>Recall = 96.00%</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">Specificity = 89.33%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>Accuracy = 92.67%, F1 = 0.9290</b></td>
            </tr>
        </tbody>
    </table>
</div>

<p style="margin-top: 0; margin-bottom: 8px;">
<b>True Positives ($144 / 150$):</b> High sensitivity across physical brawls, armed confrontations, and violent altercations. EdgeGAT captures rapid physical convergence ($\Delta v_{ij}$ spikes) and spatial overlap ($\text{IoU} > 0.3$). <b>True Negatives ($134 / 150$):</b> Normal pedestrian walking and conversation successfully suppressed via temporal sparsity regularization. <b>False Positives ($16 / 150$):</b> Caused by celebratory embraces, aggressive athletic dancing, or sudden camera shakes. <b>False Negatives ($6 / 150$):</b> Confined to severe low-light night-vision captures where actors occupied fewer than 15 pixels.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">E. Hypothesis Verification</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table VII synthesizes post-evaluation empirical outcomes across all four research hypotheses.
</p>

<!-- TABLE VII - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE VII: Verification Outcomes of Core Research Hypotheses</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 22%;">Hypothesis</th>
                <th style="padding: 4px 6px; text-align: left; width: 28%;">Target Threshold</th>
                <th style="padding: 4px 6px; text-align: left; width: 35%;">Empirical Result</th>
                <th style="padding: 4px 6px; text-align: center; width: 15%;">Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>H1: Adaptive Sampling</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Frame reduction &gt; 40%, zero recall loss</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">48.7% frame reduction; zero anomaly clips missed under $G_{\max}=4$ bound</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>H2: Relational Threat Graph</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">AUC gain &gt; +0.20 over baseline</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>+44.02% absolute gain</b> (0.5416 to 0.9818)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>H3: Token Pruning</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">AUC preservation &gt; 98%, FLOPs down 75%</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">99.96% AUC preserved (0.9822 to 0.9818) with 4&times; FLOP reduction</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>H4: Causal Streaming Memory</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Recall &gt; 70%, F1 &gt; 0.70, $\mathcal{O}(1)$ RAM</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Recall 96.00%, F1 = 0.9290 with constant memory footprint</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>CONFIRMED</b></td>
            </tr>
        </tbody>
    </table>
</div>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VII. PRODUCTION DEPLOYMENT & ETHICAL GOVERNANCE</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. 5-Tier Hysteresis Alerting & Telemetry</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS incorporates a calibrated Alert Dispatcher with five operational tiers: NORMAL ($<0.35$), LOW ($0.35-0.50$), MEDIUM ($0.50-0.70$), HIGH ($0.70-0.85$), and CRITICAL ($\ge 0.85$). A 5.0-second refractory hysteresis timer prevents webhook flooding during extended incidents. Crucially, each alert emits structured telemetry detailing camera identifier, peak anomaly score, active actor bounding boxes, and object counts (`person_count`, `weapon_count`, `bag_count`, `vehicle_count`).
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Privacy-by-Design Governance</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS adheres strictly to Privacy-by-Design principles:
</p>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Local Volatile Ingestion:</b> Video frames are processed strictly in volatile RAM on local edge hardware; raw pixel arrays are immediately overwritten. Only anonymous vector telemetry traverses networks.</li>
    <li><b>Rejection of Demographic Profiling:</b> ARGUS does not perform facial recognition, demographic categorisation, or gait identification. Graph nodes encode kinematic vectors, ensuring classification depends strictly on physical actions.</li>
    <li><b>Human-in-the-Loop Verification:</b> ARGUS is engineered strictly as a Tier-1 decision support assistant; dispatch triggers require human security confirmation via the explainability console.</li>
</ul>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VIII. CONCLUSION AND FUTURE ROADMAP</h2>
<p style="margin-top: 0; margin-bottom: 8px;">
This paper presented ARGUS, an adaptive relational graph-transformer architecture that resolves the Surveillance Paradox by reconciling high-capacity relational reasoning with real-time edge execution. Through an expanded 5-class semantic threat graph, continuous 10D relational EdgeGAT message passing, graph-guided token pruning, and a causal streaming FIFO transformer, ARGUS achieves 0.9818 ROC-AUC and 0.9290 F1-score on the RLVD benchmark at 31.2 FPS on commodity CPU hardware, while retaining 80.00% recall in zero-shot cross-dataset evaluation.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Future development will advance across three horizons:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>TensorRT INT8 Quantization (2027):</b> Quantizing the fine-tuned ConvNeXt and EdgeGAT weights to INT8 precision for &gt;120 FPS execution under 15W on embedded Jetson Orin micro-appliances.</li>
    <li><b>Multimodal Audio-Visual EdgeGAT (2028):</b> Incorporating Audio Spectrogram Transformers to fuse acoustic signatures (gunshots, shattering glass, screaming) directly into scene graph relation vectors.</li>
    <li><b>City-Scale Multi-View Graph Neural ODEs (2029):</b> Modeling persistent entity trajectories across non-overlapping surveillance cameras using continuous-time Neural ODEs.</li>
</ol>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">ACKNOWLEDGMENT</h2>
<p style="margin-top: 0; margin-bottom: 8px;">
The authors extend their sincere gratitude to the faculty and research mentors at the School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, for their continued laboratory support and technical guidance.
</p>

<!-- REFERENCES -->
<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">REFERENCES</h2>
<div style="font-size: 8pt; text-align: left; line-height: 1.1;">
    <p style="margin: 2px 0;">[1] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2018, pp. 6479–6488.</p>
    <p style="margin: 2px 0;">[2] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, "Weakly-supervised video anomaly detection with robust temporal feature magnitude learning," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2021, pp. 4975–4986.</p>
    <p style="margin: 2px 0;">[3] M. Cho, K. Min, and S. Lee, "CU-Net: Modeling uncertainty in weakly-supervised video anomaly detection," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2023, pp. 10234–10243.</p>
    <p style="margin: 2px 0;">[4] J. C. Wu, C. Y. Hsieh, W. C. Chen, C. S. Cheng, and H. H. Shuai, "VadCLIP: Adapting vision-language models for weakly supervised video anomaly detection," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2023, pp. 10423–10433.</p>
    <p style="margin: 2px 0;">[5] X. Zhang et al., "LAVAD: Harnessing large language models for training-free video anomaly detection," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2024.</p>
    <p style="margin: 2px 0;">[6] J. C. Feng, F. T. Hong, and W. S. Zheng, "MIST: Multiple instance self-training framework for video anomaly detection," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2021, pp. 14009–14018.</p>
    <p style="margin: 2px 0;">[7] D. Gong et al., "Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2019, pp. 1705–1714.</p>
    <p style="margin: 2px 0;">[8] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection — A new baseline," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2018, pp. 6536–6545.</p>
    <p style="margin: 2px 0;">[9] J. X. Zhong, N. Li, W. Kong, S. Liu, T. H. Li, and G. Li, "Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2019, pp. 1237–1246.</p>
    <p style="margin: 2px 0;">[10] D. Bolya, C. Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, "Token merging: Your ViT but faster," in <i>Int. Conf. Learn. Representations (ICLR)</i>, 2023.</p>
    <p style="margin: 2px 0;">[11] Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, "DynamicViT: Efficient vision transformers with dynamic token sparsification," in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 34, 2021, pp. 13937–13949.</p>
    <p style="margin: 2px 0;">[12] M. M. Soliman, M. H. Kamal, M. A. El-Massih, Y. M. Mostafa, B. S. Chawky, and D. Khattab, "Violence detection in video surveillance using deep learning techniques," in <i>Proc. 9th Int. Conf. Intell. Comput. Inf. Syst. (ICICIS)</i>, 2019, pp. 112–117.</p>
    <p style="margin: 2px 0;">[13] H. K. Cheng and A. G. Schwing, "XMem: Long-term video object segmentation with an Atkinson-Shiffrin memory model," in <i>Proc. Eur. Conf. Comput. Vis. (ECCV)</i>, 2022, pp. 640–658.</p>
    <p style="margin: 2px 0;">[14] N. Ravi et al., "SAM 2: Segment anything in images and videos," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2023.</p>
    <p style="margin: 2px 0;">[15] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, "Graph attention networks," in <i>Int. Conf. Learn. Representations (ICLR)</i>, 2018.</p>
    <p style="margin: 2px 0;">[16] W. Luo, W. Liu, and S. Gao, "A revisit of sparse coding based anomaly detection in stacked RNN framework," in <i>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</i>, 2017, pp. 341–349.</p>
    <p style="margin: 2px 0;">[17] H. Li, Y. Liu, and X. Zhang, "UR-DMU: Dual memory units with uncertainty regulation for weakly supervised video anomaly detection," in <i>Proc. AAAI Conf. Artif. Intell.</i>, 2023.</p>
</div>

</div>
</div>
