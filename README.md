# Granularity-Aware Aleatoric Uncertainty Modeling for Brain Tumor Segmentation

**Authors:** Aman Kumar (`26AI91R13`), Anmol Lokhande (`22BT3AI10`)

## Overview
Medical MRI scans exhibit significant aleatoric uncertainty due to varying intensity distributions across modalities and irregular, heterogeneous tumor boundaries. Standard deterministic segmentation models (such as baseline U-Nets) apply a hard threshold ($\hat{y} = H[\eta(x) \ge 0.5]$) to pixel-level probabilities, leading to misclassifications on ambiguous boundary pixels where $\eta(x) \approx 0.5$.

This repository implements a hybrid segmentation framework combining an **Adaptive 2D U-Net** trained with **Area Under Variance (AUV) loss weighting** and refined post-hoc using **Segmentation Granular Ball Computing (SegGBC)** to improve both boundary accuracy and probabilistic calibration on the BraTS 2020 dataset.

---

## Dataset & Splits (BraTS 2020)

* **Download Data:** Download BraTS2020 dataset from the kaggle, save in root/data directory.
* **Input Modalities:** 4 aligned structural MRI channels (`FLAIR`, `T1`, `T1ce`, `T2`) normalized to `[-1, 1]` with slice dimensions of **$240 \times 240 \times 4$**.
* **Target Masks:** 3 binary ground-truth tumor sub-region channels ($240 \times 240 \times 3$).
* **Dataset Split:**
  * **Training:** 39,990 slices (~70%)
  * **Validation:** 8,525 slices (~15%)
  * **Test:** 8,680 slices (~15%)

---

## Methodology & Pipeline Architecture

### 1. Backbone Network (Adaptive U-Net)
* Symmetric encoder-decoder architecture with channel progression $8 \rightarrow 16 \rightarrow 32 \rightarrow 64 \rightarrow 128$.
* Bottleneck feature tensor dimensions: **$[128, 15, 15]$**.
* Output layer produces 3 mask logits via a $1 \times 1$ convolution.

### 2. AUV-Weighted Training Strategy
1. **Baseline Training:** Train the U-Net for **5 epochs** using positive-weighted Binary Cross-Entropy ($w_c^+ = \text{clip}\left(\frac{N_c^-}{\max(N_c^+, 1)}, 1, 20\right)$) combined with Soft Dice loss.
2. **AUV Estimation:** Extract bottleneck feature maps on the training set, compute singular values $s_j$, and calculate feature entropy $H_c$ to derive sample-level uncertainty scores ($AUV_c$) and adaptive loss weights:
   $$w_c = \text{clip}(1 - 0.75 \cdot AUV_c, 0.25, 1)$$
3. **Fine-Tuning:** Fine-tune a copy of the baseline U-Net for **2 additional epochs** using the AUV-weighted loss. *(Note: AUV introduces zero computational overhead at inference).*

### 3. Post-Hoc Refinement (SegGBC)
1. **Feature Compression:** Apply PCA to reduce the $[128, 15, 15]$ bottleneck features to 4 principal components and append scaled spatial coordinates $(x, y) \times 0.1$ to form 225 6D feature vectors.
2. **Granular Ball Clustering:** Group spatial points into adaptive granular balls via Intuitionistic Fuzzy Set (IFS) evaluation, recursive SCM splitting ($\text{SCM} < 0.8$ and $N > 20$, up to 30 iterations), and KD-Tree center-based merging ($\text{Sim} > 0.75$).
3. **Consensus Probability Blending:** Average predicted probabilities within each granular ball ($\bar{p}_k(c)$), map back to full resolution ($240 \times 240$), and blend with raw U-Net probabilities to refine noisy boundaries.

---

## Quantitative Results (Test Set)

| Evaluation Metric | Baseline U-Net | U-Net + GBC | U-Net + AUV + GBC (Ours) |
| :--- | :---: | :---: | :---: |
| **Combined Dice (↑)** | 0.8313 | 0.8452 | **0.8635** |
| **Combined IoU (↑)** | 0.7112 | 0.7319 | **0.7598** |
| **Per-Channel Dice 1 (↑)** | 0.3556 | 0.3906 | **0.4740** |
| **Per-Channel Dice 2 (↑)** | 0.6656 | 0.6887 | **0.7121** |
| **Per-Channel Dice 3 (↑)** | 0.6286 | 0.6286 | **0.7036** |
| **BCE (↓)** | 0.016124 | 0.012457 | **0.010326** |
| **Brier Score (↓)** | 0.003879 | 0.002918 | **0.002364** |
| **ECE (↓)** | 0.004221 | 0.003892 | **0.003105** |
Results of the training results are saved in training_results.
---
## Inference
To run the inference, run inference.py from the root folder.


## Reference
* Chong, *SegGBC: Justifiable Coarse-to-Fine Granular-Ball Computing*, CVPR 2026.