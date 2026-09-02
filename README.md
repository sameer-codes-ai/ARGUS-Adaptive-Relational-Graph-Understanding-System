# ARGUS — Adaptive Relational Graph Understanding for Surveillance

ARGUS is a research-first deep learning architecture for real-time aggression and video anomaly detection in surveillance streams. It combines adaptive frame selection, object-centric dynamic scene graphs, graph-guided visual token pruning, a temporal transformer, and bounded streaming memory.

---

## Quick Start

### 1. Environment Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -v
```

### 2. Smoke Testing (Synthetic Mode)

Run a 2-epoch training and evaluation test with synthetic data (denoting anomalies as moving bright patches):

```powershell
python train.py --synthetic --epochs 2 --batch-size 4
python evaluate.py --checkpoint outputs/checkpoints/best.pt --synthetic
```

---

## Dataset Integration & Manifest Generation

ARGUS natively links with external surveillance datasets including **RealLifeViolenceDataset**, **UCF-Crime**, and preprocessed uniform clips.

### Generate CSV Manifests

Generate stratified `train`, `val`, and `test` splits across all available datasets:

```powershell
python scripts/prepare_manifests.py
```

Generated CSV manifests in `data/`:
- `data/train.csv`, `data/val.csv`, `data/test.csv` (Primary default dataset)
- `data/train_violence.csv`, `data/val_violence.csv`, `data/test_violence.csv` (RealLifeViolence dataset)
- `data/train_ucf.csv`, `data/val_ucf.csv`, `data/test_ucf.csv` (UCF-Crime dataset)
- `data/train_processed.csv`, `data/val_processed.csv`, `data/test_processed.csv` (Preprocessed 16-frame clips)
- `data/train_all.csv`, `data/val_all.csv`, `data/test_all.csv` (Combined multi-dataset split)

### Supported Data Formats

Each row in a manifest CSV has `path,label,dataset,category`. `argus.data.VideoManifest` seamlessly supports:
1. **Frame directories**: Folders containing image frames (`frame_0000.jpg`, `000001.jpg`, `*.png`, etc.)
2. **Video files**: Standard video containers (`.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`)
3. **Pre-extracted `.pt` tensors**: Tensor files shaped `(frames, 3, height, width)`

---

## Training

Train ARGUS on surveillance datasets using either YAML configuration or CLI overrides:

```powershell
# 1. Train on ALL datasets combined (100% of all surveillance data)
python train.py --manifest data/train_all.csv --val-manifest data/val_all.csv --epochs 10 --batch-size 4

# 2. Train on RealLifeViolenceDataset only
python train.py --manifest data/train_violence.csv --val-manifest data/val_violence.csv --epochs 10 --batch-size 4

# 3. Train on UCF-Crime only
python train.py --manifest data/train_ucf.csv --val-manifest data/val_ucf.csv --epochs 10 --batch-size 4
```

### Resuming / Continuing Training for More Epochs

To resume an existing checkpoint (e.g. `best.pt` or `latest.pt`) and train for **few more epochs** on all datasets:

```powershell
# Continue training for 5 more epochs on all datasets
python train.py --resume outputs/checkpoints/best.pt --manifest data/train_all.csv --val-manifest data/val_all.csv --epochs 5 --batch-size 4

# Or fine-tune with a lower learning rate and reset optimizer
python train.py --resume outputs/checkpoints/best.pt --manifest data/train_all.csv --val-manifest data/val_all.csv --epochs 5 --lr 0.00005 --reset-optimizer
```

Checkpoints and training history will be saved to `outputs/checkpoints/` (`best.pt`, `latest.pt`, `training_history.json`).

---

## Evaluation

Evaluate a trained model checkpoint against any test manifest to obtain ROC-AUC, Average Precision, F1-Score, Confusion Matrix, and FPS latency benchmarks:

```powershell
python evaluate.py --checkpoint outputs/checkpoints/best.pt --manifest data/test.csv
```

---

## Single-Video Inference & Explainability

Run end-to-end inference on any video file or folder of frames to get per-frame anomaly scores and keyframe sampling timelines:

```powershell
python infer.py --input "C:\Users\s\Desktop\AURA-Aggression-Uncovering-Response-Algorithm\datasets\RealLifeViolenceDataset\ViolenceP\V_1" --checkpoint outputs/checkpoints/best.pt
```

---

## Architecture & Modules

- **Adaptive Frame Sampler** (`argus/sampling.py`): Selects informative frames dynamically using motion magnitude, gradient change, entropy, and scene change indicators.
- **Dynamic Scene Graph Builder** (`argus/graph.py`): Builds relationally-rich spatial graphs with spatial proximity, IoU overlap, relative motion, and interaction persistence.
- **Edge-Aware Graph Attention (EdgeGAT)** (`argus/graph.py`): Encodes structural interactions with multi-head attention and edge attributes.
- **Graph-Visual Fusion & Pruner** (`argus/modules.py`): Topology-aware cross-attention and graph-guided token pruning.
- **Temporal Transformer & Memory** (`argus/modules.py`): Causal multi-head temporal attention with FIFO streaming memory.
- **Loss Functions & Metrics** (`argus/losses.py`, `argus/metrics.py`): BCE with temporal smoothness/sparsity and MIL ranking loss; ROC-AUC, mAP, and F1 metrics.
