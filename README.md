# NeuroLens

> Brain tumor MRI classification with **comparative XAI analysis**
> (Grad-CAM + LIME + SHAP). Replicates the Wong et al. (2025) methodology
> in PyTorch and extends it with the explainability layer the original
> paper lacks.

Course project for *Paradigmas de Aprendizagem de Máquina* (P6) at the
[Federal University of Paraíba (UFPB)](https://www.ufpb.br).

---

## Status

✅ **Phases 1, 2, and 3 complete.** 🔄 **Phase 4 (Gradio demo) next.**

Two architectures (VGG16 + ResNet50) trained on identical splits, then explained with three complementary XAI techniques (Grad-CAM + LIME + SHAP) over a targeted sample of glioma cases.

```
                 test accuracy        macro F1
VGG16            94.11% ± 0.56%       94.01% ± 0.59%
ResNet50         94.64% ± 0.55%       94.53% ± 0.57%   (technical tie)
```

The architectures tie on accuracy but agree on their *mistakes*: both miss the same
82.2% of gliomas (Phase 2). Phase 3's XAI analysis revealed *why*:
- On misclassified gliomas, both models **look at the ventricles** and ignore
  peripheral tumors visible in the MRI — structural inferential bias, not epistemic
  uncertainty.
- **SHAP saw the tumor signal both models discarded** — the information is there;
  the model just weights it too low.
- Even on correct predictions, LIME flags **non-cerebral tissue** as decisive —
  evidence of shortcut learning (~94% accuracy may be dataset-inflated).

See the **[Phase 3 write-up](docs/public/phases/phase-3-xai.md)** for the twelve
findings and the two thesis figures with side-by-side XAI comparisons.

---

## Documentation

Public documentation lives in [`docs/public/`](docs/public/) and grows as
each phase completes:

- **[Phase 3 — Comparative XAI (Grad-CAM + LIME + SHAP)](docs/public/phases/phase-3-xai.md)** — twelve findings, thesis figures with side-by-side comparisons, the structural bias diagnosis
- **[Phase 2 — Multi-Architecture (VGG16 vs ResNet50)](docs/public/phases/phase-2-architectures.md)** — fair comparison, side-by-side confusion matrices, the architecture-independent glioma ceiling
- **[Phase 1 — VGG16 Baseline](docs/public/phases/phase-1-vgg16-baseline.md)** — full 5-fold results, confusion matrix, Wong et al. comparison, parked improvements
- **Methodology references** (cross-cutting):
  - [Dataset](docs/public/methodology/dataset.md) — Brain Tumor MRI, splits, preprocessing
  - [Model](docs/public/methodology/model.md) — VGG16 & ResNet50, transfer learning, residual connections, 2-stage protocol
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
