# NeuroLens

> Brain tumor MRI classification with **comparative XAI analysis**
> (Grad-CAM + LIME + SHAP). Replicates the Wong et al. (2025) methodology
> in PyTorch and extends it with the explainability layer the original
> paper lacks.

Course project for *Paradigmas de Aprendizagem de Máquina* (P6) at the
[Federal University of Paraíba (UFPB)](https://www.ufpb.br).

---

## Status

🔄 **Phase 1 — VGG16 Baseline** (May 2026)

The training pipeline is complete and end-to-end validated against the
held-out test set in a single-fold sanity check:

```
test_accuracy = 0.9431
macro_f1      = 0.9421
per-class F1: glioma 0.90 | meningioma 0.93 | notumor 0.95 | pituitary 0.99
```

Full 5-fold cross-validation execution is queued
(estimated ~5h 25min on a Kaggle T4 GPU).
See the **[Phase 1 write-up](docs/public/phases/phase-1-vgg16-baseline.md)**
for full results and analysis.

---

## Documentation

Public documentation lives in [`docs/public/`](docs/public/) and grows as
each phase completes:

- **[Phase 1 — VGG16 Baseline](docs/public/phases/phase-1-vgg16-baseline.md)** — single-fold sanity check results, methodological deviations, what's pending
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
