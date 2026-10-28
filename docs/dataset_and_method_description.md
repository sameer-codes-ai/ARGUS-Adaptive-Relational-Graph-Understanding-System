# Chapter 6: Dataset Ecosystem and Data Engineering

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 6.1 Multi-Dataset Surveillance Ecosystem

Surveillance systems deployed in practical environments must generalize across distinct camera resolutions, lighting variations, motion blur, and crime typologies. ARGUS is engineered and evaluated across four surveillance corpora:

### 1. Real-Life Violence Dataset (RLVD) [12]
- **Total Clips:** 2,000 video clips (1,000 violent encounters, 1,000 non-violent actions).
- **Domain Characteristics:** Real-world footage captured by mobile devices, dashcams, and CCTV across public streets, bars, transport platforms, and corridors. Features diverse lighting, high camera shake, multi-agent brawls, and varied weapon usage.
- **Split Breakdown:** 1,400 training clips (70%), 300 validation clips (15%), 300 test clips (15%).

### 2. UCF-Crime Surveillance Dataset [1]
- **Total Videos:** 1,610 long untrimmed CCTV videos (810 Anomaly across 13 crime categories, 800 Normal).
- **Categories:** Fighting, Shooting, Robbery, Assault, Burglary, Vandalism, Arson, Road Accidents, Explosion, Stealing, Shoplifting, Abuse, Arrest.
- **Domain Characteristics:** Fixed overhead angles, low frame rates (often 8–15 FPS), heavy H.264 macroblocking, and infrared night vision.
- **Standardized Evaluation Split:** 290 unseen test clips (`data/test_ucf.csv`) utilized for zero-shot cross-dataset transfer testing.

### 3. ShanghaiTech Campus Dataset [16]
- **Total Sequences:** 437 video clips across 13 camera scenes with 130 abnormal events.
- **Domain Characteristics:** University campus pedestrian avenues with anomalies such as bicycles on walkways, skateboarding, and chasing.

### 4. Preprocessed Uniform Surveillance Manifests
- **Total Uniform Clips:** 3,940 clips normalized to 16 frames each across 12 granular categories.

---

## 6.2 Deterministic Stratified Hash Splitting

To guarantee zero data contamination across train, validation, and test splits while supporting deterministic reproducibility, ARGUS uses MD5-based stratified hashing:

$$\text{HashValue} = \text{MD5}(\text{filename}) \pmod{100}$$

Partition logic:
- $\text{HashValue} \in [0, 69] \implies \textbf{Train Split}$ ($70\%$)
- $\text{HashValue} \in [70, 84] \implies \textbf{Validation Split}$ ($15\%$)
- $\text{HashValue} \in [85, 99] \implies \textbf{Test Split}$ ($15\%$)

### Table 6.1: Manifest Partition Summary Across Datasets

| Manifest Name | Source Dataset | Total Clips | Normal ($y=0$) | Anomaly / Violent ($y=1$) | Operational Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `data/train_violence.csv` | RLVD | 1,400 | 700 (50.0%) | 700 (50.0%) | Threat-5 model training |
| `data/val_violence.csv` | RLVD | 300 | 150 (50.0%) | 150 (50.0%) | Checkpoint tuning & early stopping |
| `data/test_violence.csv` | RLVD | 300 | 150 (50.0%) | 150 (50.0%) | Primary in-domain benchmark |
| `data/test_ucf.csv` | UCF-Crime | 290 | 150 (51.7%) | 140 (48.3%) | Zero-shot cross-dataset evaluation |
| `data/test_all.csv` | Unified (RLVD + UCF) | 605 | 321 (53.1%) | 284 (46.9%) | Cross-domain joint evaluation |

---

## 6.3 Manifest File Schema

Manifest files are stored as standardized comma-separated values (CSV) with three essential columns:
```csv
video_path,label,category
datasets/violence/violent/V_102.mp4,1,Violence
datasets/violence/non_violent/NV_344.mp4,0,Normal
datasets/ucf_crime/Robbery/Robbery012_x264.mp4,1,Robbery
```

- `video_path`: Relative or absolute file path to the target video container.
- `label`: Binary classification target ($0 = \text{Normal}$, $1 = \text{Anomalous/Violent}$).
- `category`: Granular string identifier for category-level error auditing.

---

## 6.4 Data Ingestion & Preprocessing Pipeline

The PyTorch dataset loader (`VideoManifest` in `argus/data.py`) implements a robust streaming ingestion pipeline:

1. **Format Agnostic Decoding:** Supports MP4, AVI, MOV, and MKV containers using OpenCV VideoCapture.
2. **Standardized Spatial Scaling:** Resizes incoming frames to $128 \times 128$ pixels via bilinear interpolation for edge processing (configurable to $224 \times 224$ for server GPUs).
3. **Clip Temporal Sampling:** Extracts fixed sequences of $T = 16$ frames using uniform striding or adaptive keyframe selection.
4. **Tensor Normalization:** Converts pixel values to floating-point tensors normalized to $[0.0, 1.0]$.
5. **Robust Error Handling:** Corrupted or unreadable video files are caught gracefully, falling back to zero-padded tensors and issuing non-blocking warnings.
