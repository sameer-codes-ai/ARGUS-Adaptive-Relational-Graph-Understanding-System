# ARGUS System Context & Architecture Registry

> Auto-generated context snapshot for ARGUS (Adaptive Relational Graph Understanding for Surveillance).
> Location: `C:\Users\Omkar Joshi\Desktop\ARGUS\.ai\project_context.md`
> Schema Version: `threat5` | Test Suite: 58/58 Passing (100%)

## 1. System Overview & Dual-Layer Paradigm

ARGUS addresses surveillance video anomaly & aggression detection with an edge-viable, causal graph-transformer architecture:

### Inner Layer (Core Neural Reasoning)
1. **Adaptive Temporal Frame Sampler** (`argus/sampling.py`):
   - Computes dynamic importance scores based on motion velocity, gradient energy, 32-bin Shannon entropy, and scene cut proxy.
   - Enforces strict coverage bound $G_{\max} = 4$ to eliminate temporal blind spots.
   - Computes attention modulation factors $\mu_t = 1.0 + \tanh(M_t)$ for the temporal transformer.
2. **Vectorized Edge-Conditioned Graph Attention (`EdgeGAT`)** (`argus/graph.py`):
   - Pure-PyTorch implementation using scatter operations (`scatter_reduce_`, `scatter_add_`) for zero per-node loop overhead (&lt;1.5 ms per frame).
3. **Graph-Visual Gated Fusion** (`argus/modules.py`):
   - Cross-attention with learned sigmoid gating between visual tokens and graph embeddings.
4. **Graph-Guided Pruner** (`argus/modules.py`):
   - Calculates cosine similarity with graph node embeddings to preserve foreground interaction tokens ($\rho = 0.50$, $4\times$ self-attention FLOP reduction).
5. **Causal Temporal Transformer** (`argus/modules.py`):
   - Causal upper-triangular masking for strict online inference without future lookahead.
   - Dynamic attention modulation scaling transition logits.
6. **Streaming FIFO Memory** (`argus/modules.py`):
   - Bounded temporal bank ($S = 16$ clips) caching past frame representations with detached gradients ($\mathcal{O}(1)$ RAM footprint).

### Outer Layer (Production & Semantic Framework)
1. **Spatial Feature Backbone** (`argus/backbone.py`):
   - Pre-trained ConvNeXt feature extractor with built-in NativeConvNeXt pure-PyTorch fallback.
   - Multi-scale feature maps for RoI extraction.
2. **5-Class Semantic Threat Graph & Object Detector** (`argus/detector.py` & `argus/graph.py`):
   - 5 Semantic Classes: Person (0), Vehicle (1), Bag (2), Weapon (3), Other Object (4).
   - 12D Kinematic-Semantic Node Features: 6 kinematic + 5 class one-hot + 1 detection confidence.
   - 64D Bilinear RoI Visual Feature Extractor (`RoIFeatureExtractor`), yielding 76D total node representations.
   - 10D Continuous Relational Threat Edge Conditioning:
     - 4 Kinematic: Distance, IoU, Relative Velocity, Persistence.
     - 3 Broad Topology: Person-to-Person (P2P), Person-to-Object (P2O), Object-to-Object (O2O).
     - 3 Threat Indicators: Person-to-Weapon (P2W), Person-to-Bag (P2B), Weapon-to-Other (W2O).
   - YOLO11n integration (`yolo11n.pt`) with explicit COCO-to-ARGUS threat mapping.
3. **Multi-Modal Explainability Engine** (`argus/explain.py`):
   - Temporal attention map extraction.
   - GNN node attribution and relational edge highlighting.
   - Saliency heatmaps and concurrent entity counting telemetry.
4. **Real-Time Alert Dispatcher & Dashboard Interface** (`argus/alerts.py`):
   - Calibrated 5-tier severity hierarchy (NORMAL, LOW, MEDIUM, HIGH, CRITICAL).
   - Cooldown hysteresis ($\tau_{\text{cooldown}} = 5.0$s) and rate-limiting.
   - Standalone interactive HTML dashboard generation (`generate_dashboard_html`) with concurrent entity counters (`person_count`, `weapon_count`, `bag_count`, `vehicle_count`).

---

## 2. Verified Empirical Benchmark Results

Evaluated across established surveillance test splits:

| Model Configuration | Checkpoint Path | Dataset Split | ROC-AUC | AP (PR) | Accuracy | Precision | Recall | F1-Score | Speed (CPU) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inner Baseline** | `outputs/checkpoints/best.pt` | Joint Test ($N=605$) | 0.5416 | 0.4709 | 53.06% | 0.0000 | 0.0000 | 0.0000 | 29.6 FPS |
| **Outer Frozen** | `outputs/checkpoints_outer/best.pt` | Joint Test ($N=605$) | 0.5662 | 0.5101 | 53.72% | 0.5714 | 0.0563 | 0.1026 | 21.7 FPS |
| **Joint Model** | `outputs/checkpoints_joint/best.pt` | Joint Test ($N=605$) | 0.7949 | 0.7449 | 72.73% | 0.7179 | 0.6901 | 0.7038 | 13.6 FPS |
| **ARGUS Threat-5 (Zero-Shot)**| `outputs/checkpoints_rlvd_full/best.pt` | UCF-Crime Test ($N=290$) | 0.4670 | 0.4713 | 49.31% | 0.4848 | **0.8000** | 0.6038 | **33.5 FPS** |
| **ARGUS Threat-5 (Champion)** | `outputs/checkpoints_rlvd_full/best.pt` | **RLVD Test ($N=300$)** | **0.9818** | **0.9819** | **92.67%** | **0.9000** | **0.9600** | **0.9290** | **31.2 FPS** |

### Confusion Matrices:
- **RLVD Test ($N=300$):** $TP = 144, FP = 16, TN = 134, FN = 6$ (Accuracy: 92.67%, F1: 0.9290).
- **UCF-Crime Test ($N=290$):** $TP = 112, FP = 119, TN = 31, FN = 28$ (Recall: 80.00%, F1: 0.6038).

---

## 3. Academic Documentation Suite Registry
- `result_analysis.md`: Comprehensive empirical evaluation report, ablations, confusion matrices, and hardware profiling.
- `Paper.md` & `FinalPaper.md`: IEEE two-column research paper manuscript formatted for camera-ready submission.
- `ARGUS_Presentation_Deck.md`: 15-slide academic review, conference presentation, and defense slide deck.
- `docs/introduction.md`: Chapter 1 (Abstract, Problem Statement, $RQ_1-RQ_4$, $H_1-H_4$)
- `docs/01_literature_survey.md`: Chapter 2 (VAD Taxonomy, Comparative Matrix 2018-2026, Research Gaps)
- `docs/02_novelty_analysis.md`: Chapter 3 (Module-by-Module Novelty Matrix, Threat-5 Semantic Graph)
- `docs/03_research_design.md`: Chapter 4 (Theoretical Blueprints, Relational Topology Formulation)
- `docs/methodology.md`: Chapter 5 (Mathematical Formulations, EdgeGAT equations, Loss Objectives, Pseudocode)
- `docs/dataset_and_method_description.md`: Chapter 6 (Multi-Dataset Ecosystem, Stratified Hashing, Manifests)
- `docs/07_experimental_setup.md`: Chapter 7 (Hardware, Dependencies, YAML Configurations, Execution Protocols)
- `docs/08_result_analysis.md`: Chapter 8 (Detailed Empirical Results, Benchmarking, Ablations)
- `docs/09_system_architecture.md`: Chapter 9 (Production Topologies, Alert Dispatcher, Web Console)
- `docs/10_conclusion_and_future_work.md`: Chapter 10 (Synthesis, Ethical Governance, 3-Horizon Roadmap)
