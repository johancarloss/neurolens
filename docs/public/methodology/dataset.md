# Dataset — Brain Tumor MRI Classification

> Reference document — describes the dataset used across all phases of NeuroLens.
> Cross-referenced from `phases/*` whenever results are reported.

---

## Source

**Brain Tumor MRI Dataset** by Masoud Nickparvar, hosted on Kaggle:

- URL: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
- License: as published on Kaggle (open, redistributable for research)
- Curation: aggregation of three previously published datasets — figshare, SARTAJ, and Br35H — re-organized and de-duplicated by Nickparvar

The dataset is a community-curated bundle, not a clinical-grade benchmark. It is, however, the same source used by the methodology we are replicating (Wong et al. 2025), which makes results directly comparable.

---

## Composition

**7,023 grayscale MRI slices**, all 2D, across **4 mutually exclusive classes**:

| Class | Description | Training | Testing | Total |
|-------|-------------|----------|---------|-------|
| `glioma` | Tumor arising from glial cells (the supportive tissue of the brain) | 1,321 | 300 | 1,621 |
| `meningioma` | Tumor of the meninges (membranes covering the brain) | 1,339 | 306 | 1,645 |
| `notumor` | MRI scan without any visible tumor | 1,595 | 405 | 2,000 |
| `pituitary` | Tumor of the pituitary gland at the base of the brain | 1,457 | 300 | 1,757 |
| **Total** | | **5,712** | **1,311** | **7,023** |

Class balance is reasonable (no class is below 18% of the dataset), so accuracy is a meaningful metric here — see [`metrics.md`](metrics.md).

---

## Split strategy

The dataset author **already provides** a `Training/` and `Testing/` partition. We **respect that boundary**:

- `Training/` (5,712 images) — used for 5-fold stratified cross-validation
- `Testing/` (1,311 images) — held-out test set, never touched during CV

This protects against information leakage. The test set is a true generalization benchmark, evaluated only **after** model selection.

```
Training/                 → 5-fold stratified CV
├── glioma/      (1,321)     fold 0: train 80%, validate 20%
├── meningioma/  (1,339)     fold 1: train 80%, validate 20%
├── notumor/     (1,595)     ...
└── pituitary/   (1,457)     fold 4: train 80%, validate 20%

Testing/                  → Held-out evaluation only
├── glioma/      (300)
├── meningioma/  (306)
├── notumor/     (405)
└── pituitary/   (300)
```

See [`training.md`](training.md) for cross-validation details.

---

## Preprocessing pipeline

All images go through a normalization pipeline before reaching the model. Two pipelines exist:

### Training transforms (with data augmentation)

```
1. Load as PIL image, convert to torchvision tensor
2. Resize to 224 × 224 (VGG16 input resolution)
3. RandomAffine (shear up to ~11.5°)        ┐
4. RandomResizedCrop (zoom ±20%)            ├── Wong et al. augmentation
5. RandomHorizontalFlip (p = 0.5)           ┘
6. Convert to float32, scale to [0, 1]
7. Normalize with ImageNet mean/std:
     mean = [0.485, 0.456, 0.406]
     std  = [0.229, 0.224, 0.225]
```

### Evaluation transforms (deterministic, no augmentation)

```
1. Load, convert
2. Resize 224 × 224
3. Convert to float32, scale to [0, 1]
4. Normalize with ImageNet mean/std
```

The validation and test splits never see augmentation — they must measure generalization to fixed inputs.

### Why ImageNet normalization?

VGG16 (and ResNet50 in Phase 2) are pretrained on ImageNet. Their input distribution expects ImageNet normalization. Wong et al. used Keras's `rescale=1/255` (simpler [0, 1] scaling), but the more correct choice for ImageNet-pretrained backbones is the per-channel normalization. This is the only **intentional deviation** from Wong's methodology, and is documented in [phase-1](../phases/phase-1-vgg16-baseline.md#methodological-deviations).

---

## Class imbalance considerations

The class distribution is **mildly imbalanced** (max/min = 2,000 / 1,621 ≈ 1.23). This is gentle enough that:

- No class re-weighting is applied during loss computation
- Stratified sampling in CV ensures each fold preserves the global class ratio
- **Macro F1** is reported alongside accuracy, so rare-class performance is not hidden — see [`metrics.md`](metrics.md)

If imbalance were severe (e.g., 10:1), we would consider focal loss, class weights, or oversampling. Not needed here.

---

## Known limitations

These are limitations we acknowledge and discuss in the final report — they are not novel discoveries:

1. **No patient metadata**: the dataset has no demographic, scanner, or temporal information. We cannot stratify performance by patient subgroup or scanner type.
2. **Aggregated from heterogeneous sources**: figshare + SARTAJ + Br35H were originally separate datasets. Slight differences in scanner protocol, slice thickness, and orientation exist across subsets and are absorbed into the noise.
3. **2D slices only**: brain tumors are 3D structures. A single 2D slice may not show the full tumor extent. Clinical practice uses 3D volumes. Out of scope for this work.
4. **No segmentation**: classification only (which class). Segmentation (where the tumor is, pixel by pixel) is a separate task, also out of scope.

These limitations are inherent to the dataset and **shared with the baseline papers** we compare against (Wong et al. 2025, Rahman et al. 2025), so they do not affect comparison validity.

---

## Citation

If you use this dataset, please cite the original Kaggle entry:

> Nickparvar, M. (2021). *Brain Tumor MRI Dataset* [Data set]. Kaggle.
> <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
