# ARGUS System Context & Architecture Registry

> Auto-generated context snapshot for ARGUS (Adaptive Relational Graph Understanding System).
> Location: `C:\Users\s\Desktop\ARGUS\.ai\project_context.md`

## 1. System Overview & Dual-Layer Paradigm

ARGUS addresses surveillance video anomaly & aggression detection with a dual-layer architecture:

### Inner Layer (Core Neural Reasoning)
1. **Adaptive Temporal Frame Sampler** (`argus/sampling.py`):
   - Computes dynamic importance scores based on motion velocity, gradient energy, 32-bin Shannon entropy, and scene cut proxy.
   - Enforces strict coverage bound $G_{\max} = 4$ to eliminate temporal blind spots.
   - Computes attention modulation factors $\mu_t$ for the temporal transformer.
2. **Vectorized Edge-Conditioned Graph Attention (`EdgeGAT`)** (`argus/graph.py`):
   - Pure-PyTorch implementation using scatter operations (`scatter_reduce_`, `scatter_add_`) for zero per-node loop overhead.
3. **Graph-Visual Gated Fusion** (`argus/modules.py`):
   - Cross-attention with learned sigmoid gating between visual tokens and graph embeddings.
4. **Graph-Guided Pruner** (`argus/modules.py`):
   - Calculates cosine similarity with graph node embeddings to preserve foreground interaction tokens ($\rho = 0.50$).
5. **Causal Temporal Transformer** (`argus/modules.py`):
   - Causal upper-triangular masking for strict online inference.
   - Dynamic attention modulation based on spatial movement velocity and temporal entropy.
6. **Streaming FIFO Memory** (`argus/modules.py`):
   - Bounded temporal bank ($S = 16$ clips) caching past frame representations with detached gradients.

### Outer Layer (Production & Semantic Framework)
1. **Spatial Feature Backbone** (`argus/backbone.py`):
   - Pre-trained ConvNeXt feature extractor with built-in NativeConvNeXt pure-PyTorch fallback.
   - Multi-scale feature maps for RoI extraction.
2. **Semantic Scene Graph & Object Detector** (`argus/detector.py` & `argus/graph.py`):
   - Semantic typing: Person (0), Vehicle (1), Generic Object (2) with 10D node features.
   - 7D Relational Edge Conditioning:
     - Person-to-Person (P2P) interactions.
     - Person-to-Object (P2O) interactions.
     - Object-to-Object (O2O) interactions.
   - Pure-PyTorch Bilinear RoI Feature Extractor (`RoIFeatureExtractor`).
3. **Multi-Modal Explainability Engine** (`argus/explain.py`):
   - Temporal attention map extraction.
   - Gradient-based GNN node attribution.
   - Saliency heatmaps painted over bounding boxes.
4. **Real-Time Alert Dispatcher & Dashboard Interface** (`argus/alerts.py`):
   - Severity ranking (NORMAL, LOW, MEDIUM, HIGH, CRITICAL).
   - Cooldown hysteresis ($\tau_{\text{cooldown}} = 5.0$s) and rate-limiting.
   - Standalone interactive HTML dashboard generation (`generate_dashboard_html`).

## 2. Verified Empirical Benchmark Results (`checkpt1.txt`)

Evaluated on unified cross-domain test manifest `data/test_all.csv` ($N = 605$ test clips):

| Model | Checkpoint | ROC-AUC | AP (PR) | Accuracy | Precision | Recall | F1-Score | Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inner Baseline** | `outputs/checkpoints/best.pt` (Ep 1) | 0.5416 | 0.4709 | 53.06% | 0.0000 | 0.0000 | 0.0000 | 29.6 FPS |
| **Outer Frozen** | `outputs/checkpoints_outer/best.pt` (Ep 7)| 0.5662 | 0.5101 | 53.72% | 0.5714 | 0.0563 | 0.1026 | 21.7 FPS |
| **Joint E2E** | `outputs/checkpoints_joint/best.pt` (Ep 12)| **0.7949** | **0.7449** | **72.73%** | **0.7179** | **0.6901** | **0.7038** | **13.6 FPS** |

Confusion Matrix (Joint E2E): $TP = 196, FP = 77, TN = 244, FN = 88$.

## 3. Academic Documentation Suite Registry
- `docs/introduction.md`: Chapter 1 (Abstract, Problem Statement, $RQ_1-RQ_4$, $H_1-H_4$)
- `docs/01_literature_survey.md`: Chapter 2 (VAD Taxonomy, Comparative Matrix, Research Gaps)
- `docs/02_novelty_analysis.md`: Chapter 3 (10-Module Novelty Breakdown, Failure Mode Analysis)
- `docs/03_research_design.md`: Chapter 4 (Mathematical Blueprints, Ablation Status)
- `docs/methodology.md`: Chapter 5 (Formulations, EdgeGAT equations, Pseudocode)
- `docs/dataset_and_method_description.md`: Chapter 6 (Ecosystem, Manifests, Ingestion Pipeline)
- `docs/07_experimental_setup.md`: Chapter 7 (Hardware, Dependencies, YAMLs, Protocols)
- `docs/08_result_analysis.md` & `result_analysis.md`: Chapter 8 (Empirical Analysis, Literature Benchmarking)
- `docs/09_system_architecture.md`: Chapter 9 (Production Topologies, Streaming CLI, Alert Dispatcher)
- `docs/10_conclusion_and_future_work.md`: Chapter 10 (Synthesis, Ethics, 5-Year Roadmap)
