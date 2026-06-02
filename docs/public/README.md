# NeuroLens — Public Documentation

This folder contains the public-facing documentation that grows alongside the project. It is intended for:

- **Course evaluators** following progress phase by phase
- **Visitors of the GitHub repository** who want to understand the work
- **Future contributors** who need to understand the methodology before touching code

The private working notes (decisions, drafts, retrospective summaries, planning notes) live in `docs/private/` and are gitignored.

---

## Project phases

The project is delivered in 6 phases. Each phase produces a milestone write-up here as it completes.

| Phase | Topic | Status | Write-up |
|-------|-------|--------|----------|
| 0 | Setup & Infrastructure | ✅ Complete | _coming next_ |
| 1 | VGG16 Baseline (Wong et al. replication) | ✅ Complete — 94.11% ± 0.56% (5-fold) | [`phases/phase-1-vgg16-baseline.md`](phases/phase-1-vgg16-baseline.md) |
| **2** | **Multi-architecture (ResNet50)** | 🔄 Next | — |
| 3 | XAI analysis (Grad-CAM + LIME + SHAP) | ⏸ Not started | — |
| 4 | Gradio demo | ⏸ Not started | — |
| 5 | Polish & delivery | ⏸ Not started | — |

---

## Methodology reference

Cross-cutting concepts are documented once in `methodology/` and referenced from each phase write-up. These do not change as the project progresses — they are the canonical descriptions.

| Topic | Reference doc |
|-------|---------------|
| Dataset (Brain Tumor MRI, splits, preprocessing) | [`methodology/dataset.md`](methodology/dataset.md) |
| Clinical context (what each tumor is, how it appears on MRI) | [`methodology/clinical-context.md`](methodology/clinical-context.md) |
| Model (VGG16, transfer learning, 2-stage protocol) | [`methodology/model.md`](methodology/model.md) |
| Training (5-fold CV, hyperparameters, dual-write tracking) | [`methodology/training.md`](methodology/training.md) |
| Evaluation metrics (accuracy, F1, confusion matrix) | [`methodology/metrics.md`](methodology/metrics.md) |

---

## How to read this documentation

Start with the [Phase 1 write-up](phases/phase-1-vgg16-baseline.md) — it links to every methodology document at the relevant points. Reading them in the order they are referenced gives a natural top-down tour of the project.

For the top-level project overview, return to the root [`README.md`](../../README.md).
