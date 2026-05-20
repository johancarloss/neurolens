# 🧠 NeuroLens

> Brain tumor MRI classification with **comparative XAI analysis**
> (Grad-CAM + LIME + SHAP). Replicates Wong et al. (2025) methodology
> and extends it with the explainability layer the original paper lacks.

## Status

🚧 **Phase 0 — Setup & Infrastructure** (May 2026)

Currently building the development pipeline: package structure,
PostgreSQL schema, Weights & Biases tracking, Kaggle Kernels integration.
Model training and XAI analysis will follow in subsequent phases.

## Stack

- **Python 3.12** + [uv](https://docs.astral.sh/uv/) (package manager with `exclude-newer` supply-chain defense)
- **PyTorch 2.10** + torchvision (deep learning core)
- **grad-cam**, **lime**, **shap** (three XAI techniques compared)
- **Gradio** (interactive demo)
- **Weights & Biases** (experiment tracking, shareable dashboards)
- **PostgreSQL** (durable source of truth — dual-write with W&B)
- **Kaggle Kernels** with T4 x2 (free GPU compute)

## Quick start

```bash
git clone https://github.com/johancarloss/neurolens.git
cd neurolens
uv sync --extra dev
uv run pytest
```

Detailed setup (PostgreSQL on a VPS, W&B account, Kaggle Secrets) is
documented in the project blueprint — public guide coming as the
project matures.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar — 7,023 images across 4 classes (glioma,
meningioma, pituitary tumor, no tumor).

## Acknowledgments

This work is methodologically inspired by:

- Wong, Y., et al. (2025). *Classifying Brain Tumors on Magnetic
  Resonance Imaging by Deep Learning Techniques.* PLOS ONE.
- Rahman, A., et al. (2025). *Enhanced MRI brain tumor detection and
  classification via deep learning.* Scientific Reports.

Built as part of the *Paradigmas de Aprendizagem de Máquina* course
at the [Federal University of Paraíba (UFPB)](https://www.ufpb.br).

## License

MIT — see [LICENSE](LICENSE) for details.
