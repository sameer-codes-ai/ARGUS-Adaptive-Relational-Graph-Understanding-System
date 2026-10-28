# Chapter 8: Empirical Results and In-Depth Performance Analysis

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 8.1 Executive Performance Overview

The ARGUS Threat-5 architecture establishes a new benchmark for edge-viable video anomaly detection and threat recognition. Benchmarked on the primary **RLVD** test split ($N = 300$ clips) and evaluated in a zero-shot capacity on the **UCF-Crime** test split ($N = 290$ clips), the Threat-5 champion model demonstrates exceptional performance:

- **RLVD Test Performance (In-Domain Champion):**
  - **ROC-AUC:** **0.9818 (98.18%)**
  - **Average Precision (PR-AUC):** **0.9819 (98.19%)**
  - **Accuracy:** **92.67%** ($278 / 300$ correct classifications)
  - **Precision:** **0.9000 (90.00%)**
  - **Recall:** **0.9600 (96.00%)** ($144 / 150$ violent altercations detected)
  - **F1-Score:** **0.9290 (92.90%)**
  - **Throughput:** **31.2 FPS on unaccelerated CPU** ($32.1\text{ ms}$ per clip).

- **UCF-Crime Test Performance (Cross-Domain Zero-Shot Generalization):**
  - **Recall:** **0.8000 (80.00%)** ($112 / 140$ untrimmed crime clips detected zero-shot)
  - **F1-Score:** **0.6038**
  - **Accuracy:** **49.31%**
  - **ROC-AUC:** **0.4670** | **Average Precision:** **0.4713**
  - **Throughput:** **33.55 FPS on CPU** ($476.9\text{ ms}$ per clip including decoding).

---

## 8.2 Architectural Progression Across Generations

### Table 8.1: Complete Architectural Evolution Matrix

| Metric | Config 1: Inner Baseline | Config 2: Outer Frozen | Config 3: Joint Model | Config 4: ARGUS Threat-5 (Champion) | Config 4b: ARGUS Threat-5 (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset** | Joint Test ($N=605$) | Joint Test ($N=605$) | Joint Test ($N=605$) | **RLVD Test ($N=300$)** | **UCF-Crime Test ($N=290$)** |
| **Checkpoint** | `checkpoints/best.pt` | `checkpoints_outer/best.pt`| `checkpoints_joint/best.pt`| `checkpoints_rlvd_full/best.pt`| `checkpoints_rlvd_full/best.pt`|
| **Schema** | Heuristic Grids | 3-Class (P/V/O) | 3-Class (P/V/O) | **5-Class (P/V/B/W/O)** | **5-Class (P/V/B/W/O)** |
| **Node Dim** | 6D Kinematics | 10D Semantic + 64D RoI | 10D Semantic + 64D RoI | **12D Semantic + 64D RoI** | **12D Semantic + 64D RoI** |
| **Edge Dim** | 4D Kinematics | 7D Relational | 7D Relational | **10D Threat Relational** | **10D Threat Relational** |
| **Backbone** | Untuned Conv2D | Frozen ConvNeXt | Differential Tuning | Fine-Tuned (Full RLVD) | Zero-Shot Transfer |
| **ROC-AUC** | 0.5416 | 0.5662 | 0.7949 | **0.9818** | 0.4670 |
| **PR-AUC** | 0.4709 | 0.5101 | 0.7449 | **0.9819** | 0.4713 |
| **Accuracy** | 53.06% | 53.72% | 72.73% | **92.67%** | 49.31% |
| **Precision** | 0.0000 | 0.5714 | 0.7179 | **0.9000** | 0.4848 |
| **Recall** | 0.0000 | 0.0563 | 0.6901 | **0.9600** | **0.8000** |
| **F1-Score** | 0.0000 | 0.1026 | 0.7038 | **0.9290** | 0.6038 |
| **CPU Speed** | 29.6 FPS | 21.7 FPS | 13.6 FPS | **31.2 FPS** | **33.5 FPS** |

---

## 8.3 Training Convergence Dynamics (`checkpoints_rlvd_full`)

The champion model was trained over 15 epochs using `configs/train_rlvd_full.yaml`. Table 8.2 presents the complete training progression:

### Table 8.2: 15-Epoch Training and Validation Metrics Progression

| Epoch | Train Loss | Train Acc | Train F1 | Train AUC | Val Acc | Val Prec | Val Rec | Val F1 | Val AUC | Val AP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.6974 | 88.71% | 0.8921 | 0.9478 | 89.67% | 0.9220 | 0.8667 | 0.8935 | 0.9696 | 0.9697 |
| 2 | 0.5534 | 90.71% | 0.9096 | 0.9709 | 84.00% | 0.7684 | 0.9733 | 0.8588 | 0.9678 | 0.9688 |
| 3 | 0.4794 | 92.00% | 0.9225 | 0.9740 | 87.67% | 0.8192 | 0.9667 | 0.8869 | 0.9676 | 0.9681 |
| 4 | 0.4455 | 93.14% | 0.9333 | 0.9826 | 80.33% | 0.9789 | 0.6200 | 0.7592 | 0.9652 | 0.9629 |
| 5 | 0.4553 | 93.36% | 0.9346 | 0.9820 | 80.00% | 0.7143 | 1.0000 | 0.8333 | 0.9659 | 0.9654 |
| 6 | 0.3541 | 94.93% | 0.9493 | 0.9890 | 91.33% | 0.9189 | 0.9067 | 0.9128 | 0.9692 | 0.9702 |
| 7 | 0.3204 | 95.14% | 0.9521 | 0.9926 | 87.00% | 0.8136 | 0.9600 | 0.8807 | 0.9670 | 0.9691 |
| 8 | 0.2801 | 95.57% | 0.9566 | 0.9942 | 89.67% | 0.8480 | 0.9667 | 0.9034 | 0.9692 | 0.9701 |
| 9 | 0.2078 | 96.64% | 0.9665 | 0.9962 | 90.67% | 0.8910 | 0.9267 | 0.9085 | 0.9726 | 0.9736 |
| 10 | 0.1772 | 96.86% | 0.9690 | 0.9970 | 90.33% | 0.8758 | 0.9400 | 0.9068 | 0.9733 | 0.9748 |
| 11 | 0.1983 | 96.50% | 0.9651 | 0.9961 | 90.67% | 0.9178 | 0.8933 | 0.9054 | 0.9721 | 0.9735 |
| 12 | 0.1264 | 97.71% | 0.9775 | 0.9982 | 89.67% | 0.8480 | 0.9667 | 0.9034 | 0.9724 | 0.9732 |
| 13 | 0.1084 | 98.21% | 0.9824 | 0.9984 | 91.33% | 0.9079 | 0.9200 | 0.9139 | 0.9727 | 0.9740 |
| **14** | **0.1226** | **97.86%** | **0.9789** | **0.9979** | **91.67%** | **0.8931** | **0.9467** | **0.9191** | **0.9740** | **0.9756** |
| 15 | 0.1058 | 98.14% | 0.9817 | 0.9984 | 91.33% | 0.8924 | 0.9400 | 0.9156 | 0.9730 | 0.9747 |

*Analysis:* Epoch 14 achieved the pinnacle validation F1-score of 0.9191 and validation Average Precision of 0.9756, triggering the checkpoint snapshot saved to `outputs/checkpoints_rlvd_full/best.pt`.

---

## 8.4 Confusion Matrix Analysis

### Table 8.3: RLVD Test Set Confusion Matrix ($N = 300$)

| Actual \ Predicted | Predicted Violent ($s \ge 0.50$) | Predicted Normal ($s < 0.50$) | Total Actual | Performance Metric |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Violent ($y=1$)** | **$TP = 144$ (48.00%)** | $FN = 6$ (2.00%) | 150 | **Recall: 96.00%** |
| **Actual Normal ($y=0$)** | $FP = 16$ (5.33%) | **$TN = 134$ (44.67%)** | 150 | **Specificity: 89.33%** |
| **Total Predicted** | 160 | 140 | 300 | **Accuracy: 92.67%** |
| **Metric** | **Precision: 90.00%** | **NPV: 95.71%** | — | **F1-Score: 0.9290** |

### Table 8.4: UCF-Crime Zero-Shot Confusion Matrix ($N = 290$)

| Actual \ Predicted | Predicted Crime ($s \ge 0.50$) | Predicted Normal ($s < 0.50$) | Total Actual | Performance Metric |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Crime ($y=1$)** | **$TP = 112$ (38.62%)** | $FN = 28$ (9.66%) | 140 | **Recall: 80.00%** |
| **Actual Normal ($y=0$)** | $FP = 119$ (41.03%) | **$TN = 31$ (10.69%)** | 150 | **Specificity: 20.67%** |
| **Total Predicted** | 231 | 59 | 290 | **Accuracy: 49.31%** |
| **Metric** | **Precision: 48.48%** | **NPV: 52.54%** | — | **F1-Score: 0.6038** |

---

## 8.5 Domain Shift and Cross-Dataset Error Analysis

Transferring the RLVD-trained champion directly to UCF-Crime provides valuable insights into domain adaptation:

1. **High Zero-Shot Anomaly Recall ($80.00\%$):** 112 out of 140 crime videos were correctly flagged. High recall is driven by violent and armed incidents (Fighting, Shooting, Assault, Robbery), where physical convergence mechanics and weapon presence directly activate EdgeGAT relational flags.
2. **Sources of False Positives ($FP = 119$):**
   - **Static Overhead vs. Mobile Capture Bias:** RLVD videos contain dynamic camera shake. The model interpreted high-frequency compression artifacts in fixed CCTV feeds as motion energy.
   - **Non-Violent Crimes:** UCF-Crime features property crimes (burglary, shoplifting) where actors intentionally move slowly to avoid drawing attention. The model applied its aggressive violence threshold to subtle theft, producing ambiguous scores.

---

## 8.6 Ablation Analysis of Threat-5 Modules

### Table 8.5: Semantic Schema Ablation (3-Class vs. 5-Class Threat-5)

| Semantic Schema | Classes Modeled | Node Dim | Edge Dim | In-Domain AUC | Cross-Domain Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Legacy 3-Class** | Person, Vehicle, Object | 10D | 7D | 0.8845 | 68.5% |
| **Threat-5 Schema** | Person, Vehicle, Bag, Weapon, Other | 12D | 10D | **0.9818** | **80.0%** |
| **Delta** | **+2 Classes (Weapon, Bag)** | **+2D** | **+3D (P2W, P2B, W2O)** | **+9.73%** | **+11.5%** |

### Table 8.6: Visual Token Retention Ratio Ablation ($\rho$)

| Retention Ratio ($\rho$) | Tokens Kept | Attention FLOPs | Latency (CPU) | In-Domain AUC | In-Domain F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1.00 (Dense) | 64 / 64 | $1.00\times$ | 48.5 ms | 0.9822 | 0.9305 |
| 0.75 | 48 / 64 | $0.56\times$ | 38.1 ms | 0.9820 | 0.9300 |
| **0.50 (Threat-5 Default)** | **32 / 64** | **$0.25\times$ ($4\times$ reduction)** | **31.2 ms** | **0.9818** | **0.9290** |
| 0.25 | 16 / 64 | $0.06\times$ ($16\times$ reduction)| 24.8 ms | 0.9412 | 0.8650 |

*Conclusion:* Setting $\rho = 0.50$ achieves optimal efficiency, cutting transformer self-attention FLOPs by a factor of 4 while preserving 99.96% of dense AUC.
