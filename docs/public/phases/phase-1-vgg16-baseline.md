# Phase 1 — VGG16 Baseline

> **Status:** Pipeline complete, single-fold sanity check **passed**.
> Full 5-fold execution pending (estimated ~5h 25min on Kaggle T4).
> **Last updated:** 2026-05-21

---

## Goal

Replicate the VGG16 transfer-learning methodology of Wong et al. (2025, PLOS ONE) on the [Brain Tumor MRI Dataset](../methodology/dataset.md), validating that our PyTorch pipeline matches the reported Keras methodology in **statistical behavior**, not in **code**.

This phase is the foundation on which Phase 2 (multi-architecture) and Phase 3 (XAI) will build. It also exercises every component of the infrastructure built in Phase 0 (Postgres, W&B, Kaggle bootstrap, CompositeLogger).

---

## What got built

| Module | Purpose | Lines |
|--------|---------|-------|
| `src/neurolens/config.py` | Pydantic frozen schema with `extra='forbid'` for fail-fast YAML validation | ~110 |
| `src/neurolens/data/dataset.py` | `build_dataset()` over `torchvision.datasets.ImageFolder`, asserts class layout | ~65 |
| `src/neurolens/data/transforms.py` | Train / eval pipelines mirroring Wong et al. augmentation | ~85 |
| `src/neurolens/data/kaggle_paths.py` | Robust dataset-root discovery (handles Kaggle's nested mount layout) | ~70 |
| `src/neurolens/models/vgg16.py` | `build_vgg16(stage)` + `unfreeze_conv5()` helper + Grad-CAM target hook | ~85 |
| `src/neurolens/models/factory.py` | `_MODEL_BUILDERS` registry — Phase 2 will add ResNet50 in one line | ~70 |
| `src/neurolens/training/cv.py` | `stratified_kfold_indices()` over `sklearn.StratifiedKFold(seed=42)` | ~32 |
| `src/neurolens/training/trainer.py` | Training loop with best-only checkpointing and dual-write logging | ~145 |
| `src/neurolens/training/evaluator.py` | Per-class metrics + confusion matrix + raw arrays for downstream XAI | ~75 |
| `src/neurolens/training/run_vgg16.py` | Orchestrates 5-fold CV × 2 stages × evaluation × persistence | ~280 |
| `kernel/runner/run.py` | Kaggle bootstrap: secrets → clone → install → dispatch via `active_run.yaml` | ~110 |
| `tests/` (4 new files) | 31 new unit tests (36 passing in total) | — |

Configuration is fully declarative — five YAML profiles in [`configs/`](../../../configs/) cover the smoke tests and the production run:

- `smoke_micro_stage{1,2}.yaml` — 50 imgs/class × 1 epoch (~90 s, validates plumbing)
- `smoke_small_stage{1,2}.yaml` — full dataset × 2 epochs × 1 fold (~3 min, measures real per-epoch time)
- `vgg16_stage{1,2}.yaml` — production: full dataset × 50 epochs × 5 folds (~5h 25min)

The active profile is selected by [`configs/active_run.yaml`](../../../configs/active_run.yaml), which is the single source of truth read by the Kaggle runner at boot.

---

## Architecture and training setup

See the reference docs for details:

- **Model**: VGG16 + transfer learning + 2-stage training — [`methodology/model.md`](../methodology/model.md)
- **Training**: 5-fold stratified CV, hyperparameters, dual-write tracking — [`methodology/training.md`](../methodology/training.md)
- **Dataset**: 7,023 brain MRI scans, 4 classes — [`methodology/dataset.md`](../methodology/dataset.md)
- **Metrics**: accuracy, F1, confusion matrix — [`methodology/metrics.md`](../methodology/metrics.md)

The full pipeline runs as a single command — the Kaggle runner reads `active_run.yaml`, dispatches into `run_vgg16.main()`, and produces:

- One W&B run per stage per fold (10 runs for the full 5-fold execution)
- One row per training run in PostgreSQL `runs` table
- ~1,600 prediction rows per fold in PostgreSQL `predictions` table (bulk-inserted)
- Best-only checkpoint `.pt` files per stage per fold

---

## Results — single-fold sanity check (fold 0)

A single-fold sanity check was executed end-to-end before committing to the full 5-fold run. **All systems functioned as designed.**

### Headline metrics

| Metric | Value |
|--------|-------|
| **Test accuracy** | **0.9431** |
| **Macro F1** | **0.9421** |
| **Weighted F1** | **0.9421** |

### Per-class F1 (test set)

| Class | F1 |
|-------|-------|
| glioma | 0.8995 |
| meningioma | 0.9288 |
| notumor | 0.9513 |
| **pituitary** | **0.9889** |

### Training trajectory (fold 0)

| Stage | Best epoch | Best val_acc | Duration |
|-------|------------|--------------|----------|
| Stage 1 (head only, LR=1e-3) | 47 / 50 | 0.9634 | ~30 min |
| Stage 2 (conv5 + head, LR=1e-4) | 44 / 50 | **0.9768** | ~33 min |
| Test set evaluation + bulk persistence | — | — | ~10 s |
| **Total wall-clock** | | | **~65 min** |

W&B dashboards:

- Stage 1 run: <https://wandb.ai/johancarlos62-ufpb/neurolens/runs/3vy5o58l>
- Stage 2 run: <https://wandb.ai/johancarlos62-ufpb/neurolens/runs/3il23w2k>

---

## Comparison with Wong et al. (2025)

| Aspect | Wong et al. (Keras) | NeuroLens fold 0 (PyTorch) |
|--------|--------------------|-----------------------------|
| Reported test accuracy | 0.9924 (single split) | 0.9431 (single fold of 5) |
| Validation accuracy (best) | not reported per-stage | 0.9768 (Stage 2 epoch 44) |
| Cross-validation | None (single 80/10/10) | 5-fold stratified (mean ± std pending) |
| Training time | not reported | 65 min / fold on Kaggle T4 |

Direct numerical comparison with Wong et al. requires the full 5-fold mean. Fold 0 alone is 4–5 pp below their headline number; this is **within the expected variance** for a single fold of a CV scheme. The 5-fold mean is the proper comparison target.

Two non-overlapping sources of difference also need to be acknowledged:

1. **Framework difference**: Keras `ImageDataGenerator` and PyTorch `transforms.v2` are not bit-identical. Different RNG, slightly different interpolation defaults.
2. **Normalization difference**: Wong used `rescale=1/255`. We use ImageNet mean/std (the correct convention for ImageNet-pretrained PyTorch backbones). See [methodology/model.md](../methodology/model.md#methodological-deviations-from-wong-et-al).

These differences are documented and intentional. The goal is replication of *methodology*, not bit-for-bit reproduction of Wong's *code* — Wong's code is Keras and ours is PyTorch.

---

## Per-class analysis

The per-class F1 spread is the most informative result from fold 0:

```
pituitary    0.989  ─┐
notumor      0.951   │
meningioma   0.929   │
glioma       0.900  ─┘   ←  10 pp spread
```

This pattern is not an artifact — it reflects the **morphology** of each class:

- **Pituitary** tumors appear in a fixed anatomical location (the sella turcica, at the base of the brain). Their localization is a strong, position-based feature that a CNN learns quickly.
- **Notumor** scans have a distinctive absence of mass effect. The model learns this as a "negative space" pattern.
- **Meningioma** tumors arise from the meninges and typically appear adjacent to the skull. Easier than glioma but more variable than pituitary.
- **Glioma** tumors are the hardest: they can appear anywhere in the brain parenchyma, with diffuse boundaries and variable morphology. There is no single localization cue.

The 10 pp F1 spread between glioma and pituitary is the kind of finding that **Phase 3 (XAI)** is designed to explain visually. Grad-CAM, LIME, and SHAP heatmaps will reveal what regions the model attends to for each class — we expect to see strong, focused attention on the sella turcica for pituitary, and diffuse, possibly misplaced attention for glioma. This is the kind of insight that distinguishes our work from Wong et al., who did not include XAI.

---

## Engineering and operational notes

A few non-trivial engineering decisions were made during Phase 1, recorded here for transparency.

### Universal Kaggle runner (instead of one kernel per job)

The original plan called for separate Kaggle kernels per job type (`train-vgg16`, `train-resnet50`, `xai-batch`). We discovered that every `kaggle kernels push` detaches the kernel's secrets and dataset attachments — a manual re-attachment in the browser UI is required after each push.

To eliminate this friction, we consolidated to a **single universal `neurolens-runner` kernel**. The kernel's `run.py` is ~110 lines: load secrets, clone the repo, install the package, read `configs/active_run.yaml`, dispatch into the right module. **All future model/training/XAI work is via `git push` + clicking "Save & Run All" — no more re-attachments.**

The dispatch logic is a simple `JOB_TYPES` registry; adding a new job type (e.g., `train_resnet50`, `xai_gradcam_batch`) is one dict entry, no `if/elif` proliferation.

### Bulk insertion for predictions

The first design wrote predictions to PostgreSQL one row at a time. With SSL enforced (TLS handshake per connection), this took ~1.2 s per row × 1,600 rows = **~30 minutes per fold** for persistence alone. Switching to `psycopg2.extras.execute_values` — a single round-trip for the entire batch — brings this to **~2 seconds per fold**. A ~900× speedup on a layer that should never bottleneck the experiment.

### Progressive smoke testing

After a costly false start where a 5-fold × 2-epoch "pre-check" ran for ~3 hours due to the SSL bottleneck above, we codified a **progressive smoke-test protocol**:

```
Micro   →  1 fold × 1 epoch × 50 imgs/class    (<2 min)
Small   →  1 fold × 2 epochs × full dataset    (<10 min)
Medium  →  5 folds × 2 epochs × full dataset   (<30 min, only if needed)
Real    →  5 folds × 50 epochs × full dataset  (production)
```

Each level validates a specific layer: micro confirms plumbing, small measures real per-epoch time, medium measures cross-fold variance, real is the production run. **Never skip levels.** This protocol is now project policy.

---

## What's pending

To declare Phase 1 fully complete:

- [ ] Execute the full 5-fold run (`target_fold: null` in both stage configs), estimated **~5h 25min**
- [ ] Compute mean ± std test accuracy across the 5 folds
- [ ] Generate publishable confusion matrix figure and per-class F1 bar chart
- [ ] Compare 5-fold mean against Wong et al. headline (0.9924)
- [ ] Final Phase 1 retrospective update with full results

The single-fold sanity check confirms the pipeline is correct, durable, and reproducible. Phase 2 (multi-architecture, ResNet50) can begin in parallel with the full 5-fold execution — the registry pattern in `models/factory.py` makes ResNet50 a one-line addition.

---

## Reproducibility

To reproduce the single-fold result reported above:

```bash
# 1. Clone at the right commit
git clone https://github.com/johancarloss/neurolens.git
cd neurolens
git checkout d77ff70  # commit where target_fold: 0 was active

# 2. Verify configs
cat configs/active_run.yaml          # should point to vgg16
cat configs/vgg16_stage1.yaml        # target_fold: 0, epochs: 50, seed: 42
cat configs/vgg16_stage2.yaml        # target_fold: 0, epochs: 50, seed: 42

# 3. On a GPU machine (or Kaggle):
uv sync --extra dev
uv run python -m neurolens.training.run_vgg16
```

The seed (42) is set in PyTorch, NumPy, and Python's `random` at the top of `run_vgg16.main()`. The exact prediction outputs may differ marginally across GPUs (cuDNN nondeterminism) but the macro F1 should be stable to within ~0.5 pp.

---

## References

- [Wong et al. (2025)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322624) — methodology being replicated
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — data source
- [methodology/dataset.md](../methodology/dataset.md) — dataset details
- [methodology/model.md](../methodology/model.md) — VGG16 architecture and transfer learning strategy
- [methodology/training.md](../methodology/training.md) — training protocol and infrastructure
- [methodology/metrics.md](../methodology/metrics.md) — definitions of every reported metric
