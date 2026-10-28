# Chapter 2: Literature Survey and Related Work

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 2.1 Taxonomy of Video Anomaly Detection (VAD)

Video Anomaly Detection has evolved across three primary paradigms:

```
Video Anomaly Detection (VAD)
 ├── 1. Unsupervised Reconstruction / Prediction
 │    ├── Autoencoders & Memory Units (MemAE, Gong et al., 2019)
 │    ├── Future Frame Prediction (Liu et al., 2018)
 │    └── Diffusion-Based Normal Generative Modeling
 ├── 2. Weakly-Supervised Multiple Instance Learning (MIL)
 │    ├── Classical MIL Ranking (Sultani et al., CVPR 2018)
 │    ├── Pseudo-Label Self-Training (MIST, Feng et al., CVPR 2021)
 │    ├── Temporal Feature Magnitude Learning (RTFM, Tian et al., ICCV 2021)
 │    ├── Uncertainty Regulation (CU-Net, Cho et al., CVPR 2023)
 │    └── Dynamic Memory Units (UR-DMU, Li et al., AAAI 2023)
 ├── 3. Vision-Language & Multimodal Models (VLMs)
 │    ├── Prompt Tuning on Frozen CLIP (VadCLIP, Wu et al., ICCV 2023)
 │    ├── Training-Free LLM/VLM Reasoning (LAVAD, Zhang et al., CVPR 2024)
 │    └── Agentic Video Anomaly Reasoning (EventVAD, 2025)
 └── 4. Relational Graph-Transformer Paradigms (ARGUS Focus)
      ├── Graph Convolutional Networks (Zhong et al., CVPR 2019)
      └── Edge-Conditioned Graph Attention (ARGUS Threat-5, 2026)
```

---

## 2.2 Deep Survey by Paradigm

### 2.2.1 Unsupervised Reconstruction Baselines
Unsupervised methods model normal surveillance footage, assuming anomalies produce high reconstruction error.
- **MemAE (Gong et al., ICCV 2019):** Uses a memory-augmented deep autoencoder to store prototypical normal patterns. Achieves 71.20% AUC on ShanghaiTech. Limitation: In unconstrained real-world CCTV, complex normal interactions (e.g., bicyclists, crowded markets) produce high reconstruction error, causing severe false alarm cascades.
- **Future Frame Prediction (Liu et al., CVPR 2018):** Trains a U-Net architecture to predict the next video frame based on optical flow and intensity constraints (72.80% AUC). Suffers from motion blur degradation in low-light surveillance.

### 2.2.2 Weakly-Supervised Multiple Instance Learning (MIL)
- **Sultani et al. (CVPR 2018):** Pioneered weakly-supervised VAD using C3D features and deep ranking loss on video bags (75.41% AUC on UCF-Crime).
- **RTFM (Tian et al., ICCV 2021):** Demonstrated that anomalous snippets exhibit larger temporal feature magnitudes in top-k instances, achieving 84.30% AUC on UCF-Crime.
- **CU-Net (Cho et al., CVPR 2023):** Modeled epistemic and aleatoric uncertainty in pseudo-labels via Bayesian deep learning (85.92% AUC).
- **UR-DMU (Li et al., AAAI 2023):** Integrated dual memory units with uncertainty regulation to filter label noise (86.97% AUC).

*Core Limitation of MIL:* All classical MIL approaches operate on holistic 3D CNN clip representations (C3D, I3D). They treat the frame as an undifferentiated pixel grid, incapable of isolating interacting agents or locating specific weapons.

### 2.2.3 Vision-Language Models (VLMs)
- **VadCLIP (Wu et al., ICCV 2023):** Leveraged frozen CLIP ViT-B/16 text-image alignments with learned prompt prefixes (84.51% AUC).
- **LAVAD (Zhang et al., CVPR 2024):** Zero-shot video anomaly detection using 7B+ parameter VLMs, achieving 87.80% AUC.
- **Limitations:** Extreme inference latencies (>500 ms/frame), massive memory requirements (>14 GB VRAM), and generative hallucinations disqualify them from real-time edge CCTV deployment.

---

## 2.3 Comparative Literature Matrix (2018–2026)

| Architecture | Year | Paradigm | Visual Backbone | Benchmark AUC (%) | Edge Viable? | Spatial Grounding | Causal Streaming |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sultani et al.** [1] | 2018 | Weakly Supervised MIL | C3D | 75.41 | Moderate (GPU) | None | No |
| **MemAE** [7] | 2019 | Unsupervised Recon. | ConvAE | 71.20 | Yes (CPU) | Heatmap | No |
| **Zhong et al.** [9] | 2019 | Graph Anomaly | C3D + GCN | 81.08 | No (GPU) | Proposals | No |
| **MIST** [6] | 2021 | Self-Guided MIL | I3D | 82.30 | Moderate (GPU) | Pseudo Masks | No |
| **RTFM** [2] | 2021 | Feature Magnitude | I3D | 84.30 | Moderate (GPU) | None | No |
| **CU-Net** [3] | 2023 | Uncertainty MIL | ViT | 85.92 | No (Server GPU) | None | No |
| **VadCLIP** [4] | 2023 | Prompt Tuning | CLIP ViT-B | 84.51 | No (Server GPU) | None | No |
| **UR-DMU** [17] | 2023 | Dynamic Memory | I3D | 86.97 | No (Server GPU) | None | No |
| **LAVAD** [5] | 2024 | Zero-Shot VLM | 7B VLM + LLM | 87.80 | No (Cloud Only) | Textual | No |
| **EventVAD** | 2025 | Agentic MLLM | 7B+ Multi-Modal | 89.40 | No (Cloud Only) | Textual | No |
| **ARGUS (Joint E2E)** | 2026 | Graph-Transformer | ConvNeXt + EdgeGAT | 79.49 | Yes (13.6 FPS CPU)| Bounding Boxes | **Yes** |
| **ARGUS Threat-5** | **2026** | **5-Class Relational**| **ConvNeXt+YOLO11+EdgeGAT**| **98.18** | **Yes (31.2 FPS CPU)**| **5-Class Telemetry** | **Yes** |

---

## 2.4 Research Gaps Addressed by ARGUS

1. **Gap 1: Absence of Fine-Grained Weapon and Luggage Relational Conditioning:** Prior graph models treat edges as binary distances without isolating weapons, luggage, or physical interaction mechanics. ARGUS Threat-5 introduces a 5-class ontology with 10D relational threat edge conditioning.
2. **Gap 2: Computational Overhead of Vision Transformers on Surveillance Feeds:** Standard self-attention incurs $\mathcal{O}(N^2)$ complexity on mostly static backgrounds. ARGUS introduces topology-aware saliency token pruning, achieving a $4\times$ FLOP reduction.
3. **Gap 3: Offline Non-Causal Assumptions:** Existing benchmarks evaluate offline models with bidirectional temporal attention. ARGUS enforces strict upper-triangular causal masking and a bounded FIFO memory buffer ($\mathcal{O}(1)$ memory).
