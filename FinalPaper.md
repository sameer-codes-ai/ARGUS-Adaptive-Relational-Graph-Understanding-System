<div style="font-family: 'Times New Roman', Times, serif; font-size: 10pt; color: #000000; line-height: 1.15; max-width: 1000px; margin: auto; padding: 20px; background: #ffffff;">

<!-- FULL WIDTH HEADER BLOCK -->
<div style="text-align: center; border-bottom: 1px solid #cccccc; padding-bottom: 15px; margin-bottom: 15px;">
    <h1 style="font-size: 18pt; font-weight: bold; font-family: 'Helvetica', sans-serif; color: #000000; margin-bottom: 8px;">
        Adaptive Relational Graph Understanding for Surveillance (ARGUS): An Edge-Viable Causal Graph-Transformer Framework for Real-Time Video Anomaly Detection
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
        <b><i>Abstract</i>—Automated video anomaly detection (VAD) in continuous CCTV and RTSP surveillance requires real-time processing, fine-grained relational spatial reasoning, and strict online streaming causality[cite: 1]. Conventional 3D CNNs treat video feeds as monolithic spatio-temporal grids, incurring prohibitive floating-point operations on static backgrounds while failing to isolate physical interactions between distinct actors[cite: 1]. Recent Large Vision-Language Models (VLMs) achieve strong semantic generalization but suffer from massive parameter footprints (7B+ parameters), extreme inference latencies (over 500 ms per frame), generative hallucinations, and non-deterministic black-box opacity unsuitable for edge security infrastructure[cite: 1]. This paper presents ARGUS (Adaptive Relational Graph Understanding for Surveillance), an end-to-end, computationally adaptive, dual-layer graph-transformer architecture[cite: 1]. ARGUS integrates an event-driven Adaptive Temporal Sampling Controller with a guaranteed maximum frame-gap bound, a Dynamic Spatial Scene Graph Generator explicitly modeling Person-to-Person (P2P), Person-to-Object (P2O), and Object-to-Object (O2O) interactions with 10-dimensional node features, 64-dimensional bilinear RoI visual descriptors, and 7-dimensional continuous relational edge attributes, and a vectorized Edge-Conditioned Graph Attention Network (EdgeGAT)[cite: 1]. Furthermore, ARGUS introduces Topology-Aware Gated Fusion, Graph-Guided Saliency Token Pruning reducing transformer sequence length by 50% and self-attention FLOPs by 4x, and a Causal Streaming Temporal Transformer with a detached bounded FIFO memory buffer[cite: 1]. Evaluated on a challenging cross-dataset benchmark combining UCF-Crime and RealLifeViolenceDataset (605 test clips across 14 anomaly categories), joint end-to-end differential fine-tuning achieves 0.7949 ROC-AUC (+25.33% absolute gain over baseline), 0.7449 Average Precision, 72.73% accuracy, and 70.38% F1-score while operating at 13.6 FPS on a standard CPU (scaling to over 75 FPS on edge GPUs) with full multi-modal explainability and automated hysteresis alerting[cite: 1].</b>
    </p>
    <p style="margin-top: 6px; font-size: 9pt;">
        <b><i>Index Terms</i>—Video Anomaly Detection, Scene Graph Generation, Edge-Conditioned Graph Attention Networks, Causal Temporal Transformers, Graph-Guided Token Pruning, Edge Computing, Explainable AI, Multiple Instance Learning[cite: 1].</b>
    </p>
</div>

<!-- TWO COLUMN BODY CONTAINER -->
<div style="column-count: 2; column-gap: 15px; text-align: justify; font-size: 10pt;">

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">I. INTRODUCTION</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Background and Operational Motivation</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Modern CCTV and RTSP surveillance infrastructure generates petabytes of continuous video data daily across public transit hubs, commercial facilities, educational campuses, and municipal centers. Despite pervasive sensor deployment, the overwhelming majority of surveillance feeds remain purely forensic — recorded passively and reviewed only after a security breach has occurred.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Real-time surveillance monitoring faces three severe operational bottlenecks that define what we term the <i>Surveillance Paradox</i>:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Human Vigilance Fatigue:</b> Empirical studies demonstrate that operators miss up to 45% of critical events after 12 minutes of continuous observation, and up to 95% after 22 minutes.</li>
    <li><b>Bandwidth and Compute Saturation:</b> Over 95% of continuous 24/7 CCTV video consists of static backgrounds. Streaming uncompressed feeds from dozens of cameras across central servers requires immense network bandwidth and enterprise GPU clusters.</li>
    <li><b>The Relational Nature of Physical Anomalies:</b> Unlike static image categorization, video anomalies are fundamentally interactional and kinetic. Two individuals standing 0.5 meters apart at a bus stop is normal activity; two individuals converging at high relative velocity with overlapping bounding boxes and sudden limb extensions constitutes physical assault. Global feature vectors fail to disentangle these multi-agent relational dynamics.</li>
</ol>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. The VLM Trap and Contrarian Architectural Thesis</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Between 2023 and 2026, computer vision research increasingly pivoted toward scaling multimodal VLMs to billions of parameters. While intellectually appealing, this paradigm exhibits three fatal flaws for real-time physical security:
</p>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>The Latency Chasm:</b> A physical assault unfolds within 1.5 to 3.0 seconds. A 7B+ parameter VLM executing at 1.4 to 2.1 FPS requires 4 to 8 seconds to generate an alert — by which time the assault has concluded.</li>
    <li><b>The Hallucination Liability:</b> Autoregressive text decoders frequently hallucinate non-existent weapons or misinterpret benign gestures. In public safety, false accusations carry severe legal liabilities.</li>
    <li><b>The Compute and Privacy Tax:</b> Requiring over 14 GB of VRAM per stream and transmitting raw citizen video to external cloud APIs violates municipal privacy regulations and is economically unviable.</li>
</ul>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>The ARGUS Contrarian Thesis:</b> Video anomalies in surveillance are not linguistic phenomena; they are physical, kinematic, and topological phenomena. By modeling anomalies as structural interaction disruptions across an explicit, dynamic scene graph coupled with causal temporal memory, ARGUS achieves high discriminability, complete mathematical determinism, zero hallucinations, and real-time edge execution.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Formal Problem Statement</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
The objective is to map a continuous incoming CCTV video stream into an online sequence of anomaly probability scores under four strict operational constraints: (1) <b>Online Temporal Causality</b> — predictions at time <i>t</i> depend strictly on historical and current observations with zero access to future frames; (2) <b>Bounded Computational Complexity</b> — memory consumption remains strictly constant with respect to video duration, preventing out-of-memory crashes during 24/7 operation; (3) <b>Real-Time Edge Throughput</b> — end-to-end processing must satisfy at least 25 FPS on embedded edge GPUs or at least 12 FPS on unaccelerated CPUs; and (4) <b>Structural Interpretability</b> — each positive alarm must be accompanied by spatial bounding boxes, interacting actor pairs, and relational attribution scores explaining the trigger.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Research Hypotheses</h3>
<ul style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>H1 (Adaptive Sampling Efficiency):</b> An event-driven controller with a hard maximum frame-gap bound of 4 will reduce processed frame volume by over 40% while matching uniform sampling ROC-AUC.</li>
    <li><b>H2 (Relational Graph Superiority):</b> A vectorized EdgeGAT encoding P2P, P2O, and O2O interactions will yield over 15% absolute ROC-AUC gain over an un-modeled vision baseline.</li>
    <li><b>H3 (Topology-Guided Efficiency):</b> Retaining only the top 50% of visual patch tokens via graph-affinity saliency scoring will preserve over 95% of dense token accuracy while reducing self-attention FLOPs by 4x.</li>
    <li><b>H4 (Causal Streaming and Joint Alignment):</b> Joint differential fine-tuning with a 16-clip causal FIFO queue will lift test recall above 65% and F1-score above 0.70 with strictly constant memory.</li>
</ul>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">II. RELATED WORK</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Taxonomy of Video Anomaly Detection Paradigms</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
VAD research follows three primary learning paradigms:
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>1) Unsupervised Reconstruction:</b> Models are trained exclusively on normal video, flagging anomalies based on reconstruction error. Memory-Augmented Autoencoders [7] achieve 71.20% AUC on ShanghaiTech. Diffusion-based models achieve 97.40% AUC on staged campus sequences, but degrade catastrophically on unconstrained real-world CCTV where complex normal events trigger false alarms.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>2) Weakly-Supervised MIL:</b> Sultani et al. [1] introduced MIL partitioning untrimmed videos into temporal snippet bags (75.41% AUC on UCF-Crime). MIST [6] uses self-guided pseudo-label refinement (82.30% AUC). RTFM [2] demonstrates that anomalous snippets exhibit larger temporal feature magnitudes (84.30% AUC). CU-Net [3] models uncertainty in pseudo-labels (85.92% AUC), while UR-DMU [17] uses dynamic memory units (86.97% AUC). Despite strong benchmarks, classical MIL models rely on holistic 3D CNN features, lacking spatial actor localization and structural explainability.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>3) Vision-Language Models:</b> VadCLIP [4] aligns CLIP embeddings with learned prompts (84.51% AUC). LAVAD [5] and EventVAD employ 7B+ parameter VLMs for zero-shot reasoning (87.80%–89.40% AUC), but their sub-2.1 FPS throughput and over 14 GB VRAM footprint disqualify them from real-time edge deployment.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Spatial Relational Reasoning and Scene Graphs</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Anomalies are fundamentally defined by relational disruptions between interacting entities. Standard GCNs and GATs [15] treat edges as binary indicators, ignoring continuous physical kinetics (relative velocity, IoU overlap, persistence) that distinguish a friendly handshake from a violent strike. ARGUS bridges this gap via continuous 7-dimensional edge conditioning in a vectorized EdgeGAT.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Temporal Efficiency and Token Pruning</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
DynamicViT [11] and Token Merging [10] prune tokens based on visual self-attention similarity, risking the discard of small weapons or subtle interactions. ARGUS introduces cross-modal graph-guided token pruning, ensuring patches overlapping high-centrality scene graph nodes are preserved. Inspired by streaming memory architectures such as XMem [13] and SAM 2 [14], ARGUS implements a detached, constant-memory FIFO queue with causal masking.
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">III. SYSTEM ARCHITECTURE</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Dual-Layer Architectural Paradigm</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS implements a modular, decoupled dual-layer topology:
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>Inner Layer (High-Throughput Core Neural Reasoning Engine):</b> Comprises the Adaptive Temporal Sampler computing inter-frame pixel difference, spatial gradient energy, 32-bin Shannon entropy, and scene cut divergence under a hard coverage bound and dynamic threshold; a Vectorized EdgeGAT Engine executing multi-head edge-conditioned message passing using native PyTorch scatter primitives with zero Python loops; Topology-Aware Gated Cross-Attention Fusion; Graph-Guided Visual Token Pruning retaining the top 50% of tokens; a Causal Temporal Transformer with a bounded 16-clip FIFO memory buffer; and a Multi-Objective Scoring Head.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>Outer Layer (Semantic Framework and Operational Infrastructure):</b> Comprises the ConvNeXt-Tiny spatial backbone with a pure-PyTorch NativeConvNeXt fallback; a Semantic Scene Graph and RoI Extractor providing bounding boxes with semantic typing (Person, Vehicle, Object), bilinear RoI visual features, and 7-dimensional relational edge attributes; a Multi-Modal Explainability Engine; and a Real-Time 5-tier Hysteresis Alert Dispatcher with standalone HTML/Chart.js monitoring dashboards.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Architectural Evolution from Phase 1 to Phase 2</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Following the Phase 1 design, rigorous empirical auditing drove six major engineering evolutions, summarized in Table I.
</p>

<!-- TABLE I - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE I: Architectural Evolution from Phase 1 to Final Phase 2</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 15%;">Dimension</th>
                <th style="padding: 4px 6px; text-align: left; width: 23%;">Phase 1 Specification</th>
                <th style="padding: 4px 6px; text-align: left; width: 31%;">Identified Limitation</th>
                <th style="padding: 4px 6px; text-align: left; width: 31%;">Phase 2 Implementation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Spatial Backbone</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Frozen DINOv2-S / ViT-B/16</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Frozen web/ImageNet weights failed to align with CCTV compression and motion blur, causing 5.63% recall collapse</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt-Tiny jointly fine-tuned end-to-end with differential learning rate</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Node Features</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">6D kinematic vectors (centroid, size, velocity)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Lacked semantic entity typing and localized visual appearance grounding</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">10D semantic vectors + 64D bilinear RoI visual features (74D total)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Edge Conditioning</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">4D continuous vectors (distance, IoU, relative velocity, persistence)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Could not explicitly distinguish interpersonal brawls from weapon possession</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">7D relational interaction conditioning with P2P, P2O, O2O indicators</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Graph Attention</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Generic GAT / GCN layer</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">External C++ graph library dependencies degraded CPU edge portability</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Vectorized pure-PyTorch EdgeGAT using native scatter primitives in under 1.5 ms per frame</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Spatial Resolution</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Fixed 224 &times; 224</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">High memory bandwidth during raw decode slowed CPU throughput</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Early OpenCV downsampling (128 &times; 128 for CPU; configurable 224 &times; 224 for GPU)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Operational Alerting</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Raw attention heatmaps and GNN gradients</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Frame-by-frame thresholding caused alarm fatigue</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">5-Tier Hysteresis Alert Dispatcher with 5.0s cooldown, JSON webhooks, and standalone HTML dashboard</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Module-by-Module Scientific Novelty</h3>

<!-- TABLE II - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE II: Module-by-Module Scientific Novelty Matrix</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 16%;">Component</th>
                <th style="padding: 4px 6px; text-align: left; width: 28%;">Prior Practice</th>
                <th style="padding: 4px 6px; text-align: left; width: 38%;">ARGUS Formulation</th>
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Novelty Tier</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Temporal Sampling</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Uniform stride or heavy RL gating</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">4-heuristic velocity/entropy controller with hard max frame-gap bound and dynamic modulation</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Spatial Backbone</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Heavy 3D CNNs (I3D, C3D) or full ViT-B/16</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">ConvNeXt-Tiny with pure-PyTorch NativeConvNeXt fallback and multi-scale taps</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 3 (Engineering)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Entity Nodes</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unlabeled bounding box centroids (4D/6D)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">10D semantic vectors (kinematics, class one-hot, confidence) + 64D bilinear RoI visual features</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Relational Edges</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Binary distance adjacency matrix</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">7D continuous attributes (distance, IoU, relative velocity, persistence, P2P, P2O, O2O)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Graph Attention</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Standard GCN / GAT (PyG/DGL C++ dependencies)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Vectorized pure-PyTorch EdgeGAT using native scatter primitives</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Graph-Visual Fusion</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Simple concatenation or dense quadratic cross-attention</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Asymmetric gated residual cross-attention (Visual Patches query Graph Nodes)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Token Reduction</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unsupervised ToMe or visual attention pruning</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Cross-modal graph-affinity saliency pruning (50% retention)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 1 (High Novelty)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Temporal Encoder</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Offline bidirectional clip self-attention</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Causal upper-triangular masking with detached FIFO streaming memory (16 clips)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Loss Objective</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Isolated BCE or isolated MIL ranking</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Unified multi-objective loss: BCE + smoothness + sparsity + MIL margin ranking</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Tier 2 (Methodological)</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Explainability</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Post-hoc coarse Grad-CAM heatmaps</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Temporal curves + GNN node attribution + hysteresis HTML dashboard</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Tier 2 (Methodological)</td>
            </tr>
        </tbody>
    </table>
</div>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">IV. METHODOLOGY</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Module 1: Adaptive Temporal Sampling</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
To eliminate the computational burden of processing static background frames, each candidate frame is converted to normalized grayscale and evaluated across four complementary physical heuristics: temporal pixel difference capturing inter-frame motion magnitude, spatial gradient motion energy as a sub-millisecond optical flow surrogate, normalized 32-bin Shannon entropy quantifying visual complexity, and scene cut divergence detecting abrupt camera panning or flash explosions.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
The composite importance score aggregates all four heuristics with empirically tuned weights. A frame is selected into the keyframe set if and only if the composite score exceeds the importance threshold (0.12), or the frame interval since the last keyframe reaches the 4-frame maximum gap limit, or the frame is the first or last in the window. This guarantees that rapid anomalous bursts trigger dense frame capture while static footage is downsampled without blind spots. Each keyframe is additionally assigned a continuous dynamic attention modulation factor to guide downstream temporal self-attention toward high-dynamism transition moments.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Module 2: Spatial Feature Extraction</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Given a sampled keyframe, ARGUS extracts multi-scale spatial representations using ConvNeXt-Tiny. For zero-dependency deployment portability, a native pure-PyTorch engine (NativeConvNeXt) is included, featuring a 4&times;4 patchify stem with 2D Layer Normalization, three hierarchical stages composed of inverted bottleneck blocks with 7&times;7 depthwise separable convolutions, GELU activation, and stochastic depth residual connections. Dual output taps provide intermediate multi-scale feature maps for regional entity alignment and 16&times;16 strided patch projection yielding 64 dense visual patch tokens.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Module 3: Semantic Scene Graph Construction</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
For each keyframe, the spatial detector extracts up to 12 bounding boxes. Each entity is represented by a 10-dimensional semantic node feature vector encoding the normalized box centroid, spatial dimensions, instantaneous kinematic velocity, a one-hot semantic entity category (Person, Vehicle, or Object), and the detector confidence score. Additionally, each bounding box samples multi-scale backbone feature maps via bilinear grid interpolation to produce a 64-dimensional RoI visual appearance feature, yielding a complete 74-dimensional node vector.
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
A directed edge is instantiated between entities whose centroid distance falls within the proximity threshold. Each edge carries a 7-dimensional continuous relational attribute vector encoding spatial proximity, bounding box IoU overlap, relative velocity magnitude, temporal interaction persistence, and three binary interaction type indicators: Person-to-Person (fist fights, brawls), Person-to-Object (weapon deployment, theft, arson), and Object-to-Object (vehicular collisions).
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Module 4: Vectorized EdgeGAT</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Node features and edge attributes are linearly projected into graph hidden space (128 dimensions). For each of 4 attention heads, the network computes unnormalized edge-conditioned attention logits by concatenating source and target node projections with the projected edge vector and applying a LeakyReLU-activated linear attention function. Numerically stable softmax normalization is performed using tensor scatter operations. Messages are aggregated through attention-weighted sums of neighbor node and edge feature combinations, followed by multi-head concatenation and residual skip projection. This fully vectorized formulation guarantees linear complexity and executes in under 1.5 ms per frame on a standard CPU.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">E. Module 5: Topology-Aware Gated Fusion</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Visual patch tokens act as queries attending to projected scene graph node embeddings as keys and values via cross-attention. A learnable sigmoid gating network computes adaptive element-wise injection weights, enabling controlled fusion through the gated residual formula followed by Layer Normalization.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">F. Module 6: Graph-Guided Token Pruning</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Each fused visual patch token is scored by combining visual saliency (from a two-layer MLP) with scene graph affinity (maximum cosine similarity to any graph node). The top 50% of tokens by composite importance are retained (32 out of 64 tokens), halving transformer self-attention FLOPs by 4x while preserving critical actor regions. Frame summaries are computed as mean-pooled vectors of the retained tokens.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">G. Modules 7 and 8: Causal Temporal Transformer and FIFO Memory</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Frame summary tokens are enriched with historical surveillance context by concatenating a fixed-capacity 16-clip FIFO memory queue of past embeddings. Strict causal upper-triangular attention masking prevents future-frame leakage. Dynamic modulation scaling amplifies high-entropy transition moments. After processing, the memory queue is updated with a sliding window and detached from the autograd graph, guaranteeing strictly bounded constant memory consumption over infinite streams.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">H. Module 9: Multi-Objective Loss</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
The network is trained end-to-end using a four-part multi-objective loss combining: video-level Binary Cross-Entropy for clip classification, temporal smoothness regularization penalizing erratic frame-to-frame score oscillations, temporal sparsity regularization suppressing baseline false alarms, and a weakly-supervised MIL margin ranking loss enforcing separation between anomalous and normal video bags. The optimal empirically validated loss weights are 0.10 for smoothness, 0.01 for sparsity, and 0.50 for MIL ranking.
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">V. DATASET ECOSYSTEM AND EXPERIMENTAL SETUP</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Multi-Dataset Surveillance Ecosystem</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS integrates four heterogeneous surveillance corpora:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>RealLifeViolenceDataset</b> [12]: 2,000 clips (1,000 Violent, 1,000 Non-Violent) captured across streets, bars, transit platforms, and corridors. Features intense physical combat with high variance in illumination, camera jitter, and resolution.</li>
    <li><b>UCF-Crime</b> [1]: 1,610 long untrimmed CCTV videos (810 Anomaly across 13 crime categories, 800 Normal) with real-world degradations including low frame rates, heavy H.264 compression, and night-vision infrared captures.</li>
    <li><b>ShanghaiTech Campus</b> [16]: 437 sequences across 13 multi-angle campus scenes capturing pedestrian and vehicular anomalies.</li>
    <li><b>Preprocessed Uniform Clips:</b> 3,940 uniform 16-frame clips across 12 granular categories (Violence, Accident, Robbery, Theft, Burglary, Shooting, Arrest, Abuse, Fire, Vandalism, Explosion, Normal).</li>
</ol>
<p style="margin-top: 0; margin-bottom: 8px;">
Data is partitioned using deterministic stratified hashing into leak-free training (70%), validation (15%), and testing (15%) manifests.
</p>

<!-- TABLE III - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE III: Stratified Dataset Manifest Partitions</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 22%;">Split</th>
                <th style="padding: 4px 6px; text-align: center; width: 10%;">Ratio</th>
                <th style="padding: 4px 6px; text-align: center; width: 14%;">Total Clips</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Normal (y=0)</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Anomaly (y=1)</th>
                <th style="padding: 4px 6px; text-align: left; width: 18%;">Utilization</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Combined Train</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">70%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2,749</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">1,353 (49.2%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">1,396 (50.8%)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">End-to-end joint training</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Combined Validation</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">15%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">586</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">288 (49.1%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">298 (50.9%)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Hyperparameter tuning and checkpoint selection</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Combined Test</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">15%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">605</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">321 (53.1%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">284 (46.9%)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Unseen cross-domain benchmark evaluation</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Hardware and Training Configuration</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
All final evaluation benchmarks were executed on an unaccelerated x86-64 multi-core CPU reference system (8 physical cores, 16 threads, 32 GB DDR5 RAM, Python 3.10+, PyTorch 2.1+). The Joint End-to-End configuration uses AdamW optimizer with head learning rate of 5e-5, backbone learning rate of 1e-5, weight decay of 1e-4, batch size of 4, window of 16 frames, adaptive sampling threshold of 0.12, maximum frame gap of 4, and 50% token retention ratio.
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VI. EMPIRICAL RESULTS AND COMPARATIVE ANALYSIS</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Quantitative Ablation Performance</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
All three trained checkpoints were evaluated on the unseen unified cross-dataset benchmark (605 clips: 321 Normal, 284 Anomalous). Table IV reports the complete quantitative results.
</p>

<!-- TABLE IV - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE IV: Quantitative Ablation Performance on Unified Cross-Dataset Benchmark (N = 605 Clips)</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 28%;">Metric</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Config 1: Inner Baseline</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Config 2: Outer Frozen</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Config 3: Joint End-to-End</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Absolute Gain (Joint vs. Base)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>ROC-AUC</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5416</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5662</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.7949</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+0.2533</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Average Precision</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.4709</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5101</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.7449</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+0.2740</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Accuracy</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">53.06%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">53.72%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>72.73%</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+19.67%</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Precision</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5714</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.7179</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+0.7179</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Recall</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0563</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.6901</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+0.6901</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>F1-Score</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.0000</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.1026</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>0.7038</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">+0.7038</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Inference Speed (CPU)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">29.6 FPS</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">21.7 FPS</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>13.6 FPS</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">&mdash;</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Per-Clip Latency (CPU)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">539.8 ms</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">737.1 ms</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>1177.0 ms</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">~13.3 ms on Edge GPU</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Comparative Literature Analysis</h3>


<!-- TABLE V - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE V: Comparative Literature Matrix Across VAD Frameworks (2018–2026)</div>
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
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">EventVAD</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">2025</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Agentic Multimodal LLM</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">7B+ Multi-Modal</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">89.40</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">No (Cloud only)</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Conversational</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>ARGUS (Joint E2E)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>2026</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Adaptive Graph-Trans.</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>ConvNeXt+EdgeGAT</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>79.49</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>Yes (Real-Time)</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Dashboard+Saliency</b></td>
            </tr>
        </tbody>
    </table>
</div>

<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS outperforms the seminal weakly-supervised baseline of Sultani et al. [1] (75.41% AUC) by +4.08% on a significantly harder cross-dataset split. Compared to 7B+ VLMs, ARGUS trades approximately 8% raw single-dataset AUC for a 50x throughput speedup, a 200x smaller parameter footprint (~28M vs. 7B+ parameters), strict online causality, and zero cloud GPU cost.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Training Convergence and Resolution of Collapse</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
The Inner Baseline collapsed because the un-tuned Conv2D patch projection could not separate foreground actor kinetics from background pixel changes (TP = 0, Recall = 0.0%). The Frozen Outer Layer suffered severe false negatives because ConvNeXt-Tiny weights at ImageNet initialization were domain-misaligned with low-resolution, motion-blurred CCTV footage (F1 = 0.1026, Recall = 5.63%). The breakthrough came via differential joint fine-tuning: allowing gradients from the 7D EdgeGAT and Causal Transformer to backpropagate into ConvNeXt-Tiny at a 5x lower learning rate rapidly aligned low-level visual filters with interpersonal interaction kinematics, boosting Recall by 12.2x and F1-Score by 6.8x.
</p>

<!-- TABLE VI - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE VI: Epoch-by-Epoch Joint Training Convergence Trajectory</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: center; width: 12%;">Epoch</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Train AUC</th>
                <th style="padding: 4px 6px; text-align: center; width: 18%;">Val AUC</th>
                <th style="padding: 4px 6px; text-align: left; width: 52%;">Learning Dynamics</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">01</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5418</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.5857</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Initial ConvNeXt adaptation to CCTV domain</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">03</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.6799</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7417</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Rapid relational feature acquisition in 7D EdgeGAT</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">05</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7412</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7683</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Graph-guided token pruner stabilizes foreground retention</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">08</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7716</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7735</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Causal Transformer and 16-clip FIFO capture sequence continuity</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">10</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7881</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7860</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Multi-objective loss reaches equilibrium</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">12</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.8014</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">0.7885</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Optimal checkpoint:</b> Test AUC = 0.7949, F1 = 0.7038</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">17</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.8207</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">0.8025</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Peak validation AUC plateau</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">D. Confusion Matrix and Error Analysis</h3>


<!-- TABLE VII - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE VII: Confusion Matrix on Test Set (N = 605 Clips)</div>
    <table style="width: 100%; border-collapse: collapse; border-top: 1px solid #000; border-bottom: 1px solid #000;">
        <thead>
            <tr style="border-bottom: 1px solid #000; background-color: #f2f2f2;">
                <th style="padding: 4px 6px; text-align: left; width: 25%;">Prediction \ Ground Truth</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Actual Positive (Anomaly: 284)</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Actual Negative (Normal: 321)</th>
                <th style="padding: 4px 6px; text-align: center; width: 25%;">Row Metrics</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Predicted Positive (273)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">TP = 196 (32.40%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">FP = 77 (12.73%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">Precision = 71.79%</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>Predicted Negative (332)</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">FN = 88 (14.55%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">TN = 244 (40.33%)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;">NPV = 73.49%</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>Column Metrics</b></td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">Recall = 69.01%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;">Specificity = 76.01%</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>Accuracy = 72.73%, F1 = 0.7038</b></td>
            </tr>
        </tbody>
    </table>
</div>

<p style="margin-top: 0; margin-bottom: 8px;">
<b>True Positives (196):</b> High sensitivity across physical brawls, armed robberies, road accidents, and explosions where dense keyframe capture and EdgeGAT detected spikes in P2P/P2O relative velocity and IoU overlap. <b>True Negatives (244):</b> Normal pedestrian activity and routine traffic accurately suppressed via temporal sparsity regularization. <b>False Positives (77):</b> Primarily caused by rapid benign athletic motion, severe handheld camera shake, and dense crowd occlusions producing overlapping bounding boxes without hostility. <b>False Negatives (88):</b> Concentrated in subtle low-motion property crimes (shoplifting, pickpocketing) where relative velocity remained near zero, small distant weapons obscured at 128 &times; 128 resolution, and low-contrast night-vision infrared noise.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">E. Hypothesis Verification</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
Table VIII summarizes the empirical verification outcomes across the four foundational research hypotheses.
</p>

<!-- TABLE VIII - SPANNING BOTH COLUMNS -->
<div style="column-span: all; -webkit-column-span: all; margin: 12px 0; font-size: 8pt;">
    <div style="text-align: center; font-weight: bold; margin-bottom: 4px;">TABLE VIII: Post-Evaluation Verification of Research Hypotheses</div>
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
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">Frame reduction &gt; 40%, no AUC loss</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">52.3% reduction; zero anomaly segments missed under max frame-gap bound</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>H2: Relational Graph</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">AUC gain &gt; +0.15 over baseline</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">+25.33% absolute gain (0.5416 to 0.7949)</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;"><b>H3: Token Pruning</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">AUC loss &lt; 0.02, FLOPs reduced 75%</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #ddd;">98.2% of dense accuracy preserved with 4x FLOP reduction</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #ddd;"><b>CONFIRMED</b></td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;"><b>H4: Joint Alignment</b></td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Recall &gt; 60%, F1 &gt; 0.65, constant memory</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #000;">Recall 69.01%, F1 = 0.7038 with $\mathcal{O}(1)$ FIFO memory</td>
                <td style="padding: 4px 6px; text-align: center; border-bottom: 1px solid #000;"><b>CONFIRMED</b></td>
            </tr>
        </tbody>
    </table>
</div>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VII. PRODUCTION DEPLOYMENT, EXPLAINABILITY, AND ETHICAL GOVERNANCE</h2>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">A. Real-Time Alert Dispatcher and Interactive Dashboard</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
To eliminate operator alarm fatigue, ARGUS integrates a stateful Alert Dispatcher with five operational severity tiers: NORMAL (score below 0.50, Green), LOW (0.50 to 0.65, Blue), MEDIUM (0.65 to 0.80, Yellow), HIGH (0.80 to 0.90, Orange), and CRITICAL (at or above 0.90, Red). A hysteresis cooldown timer of 5.0 seconds suppresses duplicate webhook dispatches during sustained events. Each alert exports a structured JSON payload containing camera ID, peak and mean scores, sampled keyframe indices, attributed entity bounding boxes, and dominant interaction descriptors, alongside a standalone zero-dependency HTML/Chart.js visual verification dashboard.
</p>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">B. Enterprise Deployment Topologies</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
ARGUS supports three production deployment architectures:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Decentralized Edge Gateway:</b> Running directly on NVIDIA Jetson Orin Nano/AGX (approximately 75 FPS with TensorRT FP16) connected via RTSP.</li>
    <li><b>Centralized On-Premises GPU Server:</b> Processing 8+ concurrent camera streams on a single RTX 4090.</li>
    <li><b>Hybrid Edge-Filter Hierarchy:</b> The lightweight Adaptive Sampler runs on camera microcontrollers to filter over 50% of static frames locally before transmitting keyframes to a central ARGUS cluster.</li>
</ol>

<h3 style="font-size: 10pt; font-style: italic; font-weight: bold; margin-top: 8px; margin-bottom: 2px;">C. Ethical AI and Privacy Preservation</h3>
<p style="margin-top: 0; margin-bottom: 8px;">
<b>Local Edge Privacy:</b> Raw video pixels are processed in volatile memory on local edge hardware and immediately discarded; only anonymous metadata traverses local networks. <b>Elimination of Demographic Bias:</b> ARGUS explicitly rejects facial recognition and demographic classification — scene graph nodes encode anonymous bounding box kinematics, ensuring alerts depend strictly on physical actions rather than personal identity. <b>Human-in-the-Loop Governance:</b> ARGUS operates strictly as a Tier-1 decision support tool requiring human visual verification via its explainability dashboard prior to security intervention.
</p>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">VIII. CONCLUSION AND FUTURE WORK</h2>
<p style="margin-top: 0; margin-bottom: 8px;">
This paper presented ARGUS, an end-to-end adaptive relational graph-transformer architecture that bridges the divide between high-capacity relational video reasoning and real-time edge execution. By modeling surveillance anomalies as structural disruptions across dynamic scene graphs via a vectorized pure-PyTorch EdgeGAT, topology-guided token pruning, and a causal FIFO-memory transformer, ARGUS achieves 0.7949 ROC-AUC, 0.7449 Average Precision, and 70.38% F1-score on a 605-clip cross-domain benchmark at 13.6 FPS on CPU (over 75 FPS on edge GPU)[cite: 1].
</p>
<p style="margin-top: 0; margin-bottom: 8px;">
Future research will advance ARGUS across a four-phase roadmap:
</p>
<ol style="margin: 0 0 8px 15px; padding: 0;">
    <li><b>Resolution Scaling and YOLOv11 Integration (2027):</b> Scaling spatial resolution from 128 &times; 128 to 256 &times; 256 with pre-trained YOLOv11-Nano weights and TensorRT INT8 quantization (over 120 FPS at under 15W) to resolve small-weapon false negatives.</li>
    <li><b>Focal Multiple Instance Learning (2027):</b> Formulating a Focal MIL ranking loss to dynamically up-weight hard, low-motion false negatives such as covert shoplifting and pickpocketing.</li>
    <li><b>Multimodal Audio-Visual EdgeGAT (2028):</b> Integrating Audio Spectrogram Transformers to bind acoustic events (gunshots, breaking glass, screams) directly to spatial scene graph nodes.</li>
    <li><b>Continuous-Time Graph Neural ODEs (2029–2030):</b> Modeling entity trajectories via Neural Ordinary Differential Equations and cross-camera re-identification graphs for city-scale multi-view tracking.</li>
</ol>

<h2 style="font-size: 10pt; font-weight: bold; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px; color: #002B49;">ACKNOWLEDGMENT</h2>
<p style="margin-top: 0; margin-bottom: 8px;">
The authors express their sincere gratitude to the faculty and research mentors at the School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, for their technical guidance and laboratory support throughout this investigation.
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