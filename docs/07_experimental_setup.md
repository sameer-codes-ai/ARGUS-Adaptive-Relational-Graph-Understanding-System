# Chapter 7: Experimental Setup and Configuration

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 7.1 Hardware Testbed and Runtime Environment

To substantiate claims of real-time edge viability, all latency and throughput benchmarks were executed on an unaccelerated commodity x86-64 reference architecture without external cloud acceleration:

### Table 7.1: Benchmark Hardware Specification

| Component | Specification |
| :--- | :--- |
| **Processor (CPU)** | Intel Core i7-12700H (8 Performance Cores, 4 Efficient Cores, 20 Threads) |
| **Base Clock / Boost**| 2.30 GHz / 4.70 GHz |
| **System Memory (RAM)**| 32.0 GB DDR5 @ 4800 MHz |
| **Host Operating System**| Microsoft Windows 11 Enterprise (Build 22631, 64-bit) |
| **Edge Reference Platform**| NVIDIA Jetson Orin Nano (6-core Arm Cortex-A78AE, 8GB RAM, 20W TDP) |
| **Server Reference Platform**| NVIDIA GeForce RTX 4090 (24GB GDDR6X) |

---

## 7.2 Software Dependencies and Environment

- **Python Runtime:** Python 3.12.0 (compatible with 3.10+)
- **Deep Learning Framework:** PyTorch 2.13.0+cpu
- **Computer Vision:** OpenCV (`opencv-python` 4.9.0+)
- **Mathematical Utilities:** NumPy 2.5+, SciPy 1.14+
- **Test Framework:** pytest 9.1.1 (58 passed unit and integration tests)
- **Model Checkpoint Format:** Native PyTorch safe weights dictionaries with architectural metadata validation (`validate_component_metadata`).

---

## 7.3 Threat-5 Training Configuration (`configs/train_rlvd_full.yaml`)

```yaml
data:
  train_manifest: "data/train_violence.csv"
  val_manifest: "data/val_violence.csv"
  test_manifest: "data/test_violence.csv"
  frames: 16
  image_size: 128

model:
  image_size: 128
  patch_size: 16
  dim: 256
  graph_dim: 128
  max_objects: 16
  temporal_layers: 4
  heads: 4
  keep_ratio: 0.50
  memory_size: 16
  sampling_threshold: 0.12
  max_frame_gap: 4
  use_pretrained_backbone: true
  backbone_name: "convnext_tiny"
  backbone_frozen: false
  use_semantic_detector: true
  detector_name: "yolo11n.pt"
  detector_confidence: 0.30
  semantic_schema_version: "threat5"
  node_visual_dim: 64
  allow_component_fallbacks: false
  backbone_pretrained: true
  tracking_iou_threshold: 0.10

training:
  batch_size: 4
  epochs: 15
  lr: 0.0001
  weight_decay: 0.0001
  smoothness_weight: 0.10
  sparsity_weight: 0.01
  mil_weight: 0.50
  workers: 0
  seed: 42
  output_dir: "outputs/checkpoints_rlvd_full"
```

---

## 7.4 Evaluation Metrics and Mathematical Definitions

1. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures discrimination across all decision thresholds $\tau \in [0, 1]$:
   $$\text{AUC} = \int_{0}^{1} \text{TPR}(\tau) \, d\text{FPR}(\tau)$$
2. **Average Precision (PR-AUC):** Area under the Precision-Recall curve, essential for evaluating class-imbalanced video anomaly distributions:
   $$\text{AP} = \sum_{k} (R_k - R_{k-1}) P_k$$
3. **Accuracy, Precision, Recall, and F1-Score (at calibrated $\tau = 0.50$):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **End-to-End Latency and Throughput (FPS):**
   $$\text{Throughput (FPS)} = \frac{\text{Total Processed Frames}}{\text{Total Wall-Clock Time (s)}}$$
   $$\text{Clip Latency (ms)} = \frac{\text{Wall-Clock Time (s)}}{\text{Number of Clips}} \times 1000$$
