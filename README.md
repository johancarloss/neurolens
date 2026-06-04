# NeuroLens

> Brain tumor MRI classification with **comparative XAI analysis**
> (Grad-CAM + LIME + SHAP). Replicates the Wong et al. (2025) methodology
> in PyTorch and extends it with the explainability layer the original
> paper lacks.

Course project for *Paradigmas de Aprendizagem de Máquina* (P6) at the
[Federal University of Paraíba (UFPB)](https://www.ufpb.br).

---

## Status

✅ **Phase 1 (VGG16) & Phase 2 (ResNet50) complete.** 🔄 **Phase 3 (XAI) next.**

Two architectures trained on identical 5-fold splits, evaluated on the held-out test set:

```
                 test accuracy        macro F1        time/fold
VGG16            94.11% ± 0.56%       94.01% ± 0.59%  61.7 min
ResNet50         94.64% ± 0.55%       94.53% ± 0.57%  50.2 min   (+0.52 pp, -19% time)
```

The architectures essentially tie (ResNet50 wins 4/5 folds, within the std band).
The key finding: **both miss the same 82.2% of gliomas** — the difficulty is in the
data, not the architecture.

See the **[Phase 2 write-up](docs/public/phases/phase-2-architectures.md)** for the
full comparison, side-by-side confusion matrices, and the clinical trade-off analysis;
the **[Phase 1 write-up](docs/public/phases/phase-1-vgg16-baseline.md)** for the
baseline and the Wong et al. (2025) comparison.

---

## Documentation

Public documentation lives in [`docs/public/`](docs/public/) and grows as
each phase completes:

- **[Phase 2 — Multi-Architecture (VGG16 vs ResNet50)](docs/public/phases/phase-2-architectures.md)** — fair comparison, side-by-side confusion matrices, the architecture-independent glioma ceiling
- **[Phase 1 — VGG16 Baseline](docs/public/phases/phase-1-vgg16-baseline.md)** — full 5-fold results, confusion matrix, Wong et al. comparison, parked improvements
- **Methodology references** (cross-cutting):
  - [Dataset](docs/public/methodology/dataset.md) — Brain Tumor MRI, splits, preprocessing
  - [Model](docs/public/methodology/model.md) — VGG16, transfer learning, 2-stage protocol
  - [Training](docs/public/methodology/training.md) — 5-fold CV, hyperparameters, dual-write tracking
  - [Metrics](docs/public/methodology/metrics.md) — accuracy, F1, confusion matrix

The [`docs/public/README.md`](docs/public/README.md) is the index.

---

## Stack

- **Python 3.12** + [`uv`](https://docs.astral.sh/uv/) (package manager with `exclude-newer` supply-chain defense)
- **PyTorch 2.10** + torchvision (deep learning core)
- **grad-cam**, **lime**, **shap** (three XAI techniques to be compared in Phase 3)
- **Gradio** (interactive demo, Phase 4)
- **Weights & Biases** (experiment tracking, shareable dashboards)
- **PostgreSQL** (durable source of truth — dual-write with W&B)
- **Kaggle Kernels** with T4 GPU (free compute)

---

## Quick start

```bash
git clone https://github.com/johancarloss/neurolens.git
cd neurolens
uv sync --extra dev
uv run pytest
```

Reproducing a run requires PostgreSQL credentials, a W&B API key, and
access to the Kaggle dataset. See
[`docs/public/methodology/training.md`](docs/public/methodology/training.md)
for the full reproducibility checklist.

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar — 7,200 images across 4 balanced classes (glioma,
meningioma, pituitary tumor, no tumor). Full details in
[`docs/public/methodology/dataset.md`](docs/public/methodology/dataset.md).

---

## Acknowledgments

Methodologically inspired by:

- Wong, Y., et al. (2025). *Classifying Brain Tumors on Magnetic
  Resonance Imaging by Deep Learning Techniques.* PLOS ONE.
- Rahman, A., et al. (2025). *Enhanced MRI brain tumor detection and
  classification via deep learning.* Scientific Reports.

---

## License

MIT — see [LICENSE](LICENSE) for details.
