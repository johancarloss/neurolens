# Training Protocol

> Reference document — describes how models are trained in NeuroLens.
> Covers cross-validation strategy, hyperparameters, and the dual-write
> experiment tracking layer.

---

## 5-fold stratified cross-validation

Wong et al. (2025) used a single 80/10/10 train/val/test split. We use **5-fold stratified cross-validation** instead, for stronger statistical estimation.

### What it means

The training partition (5,600 images) is divided into **5 folds of equal size**, each containing ~1,120 images. For each fold *k* ∈ {0, 1, 2, 3, 4}:

```
fold k → validation set   (~20% of training partition)
folds ≠ k → training set  (~80% of training partition)
```

Five complete training runs are executed — one per fold — and final metrics are reported as **mean ± standard deviation** across the 5 folds.

### Why "stratified"?

A naive random split could place all `pituitary` examples in a single fold, leaving other folds with none. Stratification ensures each fold preserves the **global class distribution**:

```
Global ratio:   glioma 25%  meningioma 25%  notumor 25%  pituitary 25%
Each fold:      glioma 25%  meningioma 25%  notumor 25%  pituitary 25%
```

This is implemented via `sklearn.model_selection.StratifiedKFold` with a fixed seed (42).

### Why 5 folds (and not 10)?

| Folds | Validation size | Variance estimate | Compute cost |
|-------|-----------------|-------------------|--------------|
| 5     | ~20%            | Moderate          | 5 × full training |
| 10    | ~10%            | Better            | 10 × full training (≈11h on Kaggle T4) |

The Kaggle Kernels free tier caps single sessions at **9 hours**. 5-fold fits comfortably (~5h25); 10-fold does not. The variance estimate from 5 folds is sufficient for this comparison.

### Why not the held-out test set during CV?

The `Testing/` partition (1,600 images) is **never seen** during cross-validation. It is reserved for the final evaluation **after** training is complete. This protects against any form of test-set contamination.

Implementation: [`src/neurolens/training/cv.py`](../../../src/neurolens/training/cv.py).

---

## Hyperparameters

All hyperparameters are declared in YAML configs under [`configs/`](../../../configs/). They are loaded into a frozen Pydantic model (`TrainConfig`) that rejects unknown keys — a typo in YAML fails fast rather than silently using a default.

### The mental model — descending a hill

Training searches for the bottom of a valley: the point of **lowest error**. Most hyperparameters answer a question about *how to descend*:

- **Loss** (cross-entropy) is the *height* — the error being minimized. It punishes *confident wrong* answers hardest: predicting 2% for the true class is far worse than predicting 30%.
- **Learning rate** is the *step size*. Too large overshoots the valley and oscillates; too small crawls. This is exactly why the two stages differ — Stage 1's head is randomly initialized (far up the hill) so it takes larger steps (`1e-3`); Stage 2 fine-tunes the precious pretrained conv5 (already near the bottom) with 10× smaller steps (`1e-4`) so large updates don't destroy what ImageNet learned.
- **Optimizer** (Adam) decides *how* to step — it carries momentum and adapts the step size per parameter.
- **Batch size** is how many images are averaged before each step — 32 balances a stable gradient signal against T4 memory (~175 steps per epoch).
- **Epochs** is how many full passes over the data (50). More is not always better: past a point the model memorizes the training set, which is why we keep the *best* checkpoint, not the last.
- **Dropout** and **augmentation** push the model toward a *broad* valley that generalizes, rather than a narrow ditch that only fits the training images.
- **Seed** fixes all randomness so a run is reproducible.

### Stage 1 (head only)

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Learning rate | `1e-3` | Wong et al. 2025 |
| Optimizer | Adam (default β₁=0.9, β₂=0.999, ε=1e-8) | Wong et al. 2025 |
| Loss | Cross-entropy (with logits) | Standard for multi-class |
| Batch size | 32 | Wong et al. 2025 |
| Epochs | 50 | Wong et al. 2025 |
| Image size | 224 × 224 | VGG16 input requirement |
| Augmentation | Shear 0.2, Zoom 0.2, Horizontal flip | Wong et al. 2025 |
| Dropout | 0.5 | Convention (Wong did not specify) |
| Seed | 42 | Reproducibility |

### Stage 2 (fine-tune conv5 + head)

Identical to Stage 1 **except**:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | `1e-4` (10× lower) |
| Trainable parameters | Conv5 + head (everything else frozen) |

Stage 2 starts from the best Stage-1 checkpoint of the same fold.

---

## Training loop

The training loop lives in [`src/neurolens/training/trainer.py`](../../../src/neurolens/training/trainer.py).

Two nested loops run: over **epochs** (50), and within each, over **batches** (~175). For each epoch:

1. **Train pass** — for every batch of 32 images, the same five-step cycle runs:
   `forward` (predict) → `loss` (measure error) → `backward` (compute the gradient) → `optimizer step` (nudge the weights downhill) → `zero_grad` (reset for the next batch).
   The **gradient** computed by `backward` (backpropagation) is the slope of the error surface for each of the ~138M weights — *which direction reduces the loss* — propagated from the output back to the input (hence "backward"). The optimizer then takes one step of the size set by the learning rate. Weights **are updated** here.
2. **Validation pass** — iterate the validation batches under `torch.no_grad()`. The model is in `eval` mode (dropout off, full network active), no gradient is computed, and **weights are not touched**. This pass only *measures* — computing gradients would waste memory and, worse, risk learning from data reserved for honest evaluation.
3. **Checkpoint** — if `val_acc` improved, save the state dict to `checkpoints/<arch>_fold<k>_stage<s>/best.pt`. We keep **best only** — the best epoch (not necessarily the last) is what gets evaluated on the test set.
4. **Log** — emit `train/loss`, `train/acc`, `val/loss`, `val/acc` to the dual-write tracker (see below).

The contrast between the two passes is the heart of the loop: the **train pass adjusts** the model (gradients on, dropout on, weights updated); the **validation pass only measures** (gradients off, dropout off, weights frozen).

Optionally, early stopping can be enabled (`early_stopping_patience` in config). For Phase 1, it is **disabled** to stay faithful to Wong's fixed 50-epoch schedule.

---

## Dual-write experiment tracking

A single `CompositeLogger` instance writes every metric to **three independent sinks** simultaneously. If one sink fails, the others continue — the training loop is never interrupted by logging issues.

### Sink 1 — Weights & Biases (cloud)

- Project: `johancarlos62-ufpb/neurolens`
- One W&B run per (arch, stage, fold) combination
- Captures: hyperparameters, per-epoch metrics, system metrics (GPU memory, utilization), checkpoints (as artifacts)
- Provides: shareable URL, interactive plots, run-to-run comparison, custom panels

### Sink 2 — PostgreSQL (source of truth)

- Hosted on the project VPS in a Docker container with SSL
- Schema: `experiments`, `runs`, `metrics`, `predictions`, `xai_artifacts`, `xai_comparisons`
- Per training run, persists:
  - 1 row in `runs` with config + final metrics
  - ~50 rows in `metrics` (1 per epoch × multiple metric keys)
  - 1,600 rows in `predictions` (1 per test-set image with predicted class + ground truth + probabilities + correctness)
- Bulk inserts via `psycopg2.extras.execute_values` (a single round-trip per ~1,600 predictions, vs ~1,600 SSL handshakes — see [Phase 1 retrospective](../phases/phase-1-vgg16-baseline.md))

### Sink 3 — JSONL local file

- `/kaggle/working/jsonl/<run_id>.jsonl`
- One JSON object per metric emission, append-only
- Used for fine-grained debugging when W&B and Postgres aggregates don't reveal the cause of an issue

### Why three?

- **W&B** is great for visualization but is **cloud only** — if W&B is down or the API key expires, history is gone
- **Postgres** is durable and queryable with SQL, but has no built-in plotting
- **JSONL** is local and trivially parseable; survives both W&B and Postgres outages
- Together: **resilient observability** that survives any single-sink failure

---

## Reproducibility

The project pins:

- **Python 3.12** via `pyproject.toml`
- **All dependencies** via `uv.lock` (with SHA-256 hashes; `uv` verifies on install)
- **Supply-chain defense** via `exclude-newer = "7 days"` in `pyproject.toml` (ignores packages published in the last 7 days, mitigating recent supply-chain attacks)
- **Random seeds** via the YAML configs (PyTorch + NumPy + Python's `random` set in `_set_seeds()` in [`run_vgg16.py`](../../../src/neurolens/training/run_vgg16.py))

The model weights from `torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` are version-stable across torchvision releases.

A run that completes successfully can be reproduced by:

```bash
git checkout <commit-hash>
uv sync --extra dev
# edit configs/active_run.yaml to point to the right profile, then:
# (push and re-run on Kaggle, or run locally if GPU is available)
```

---

## Compute cost

Per fold, on a Kaggle T4 GPU (free tier):

| Stage | Epochs | Time per epoch | Total |
|-------|--------|----------------|-------|
| Stage 1 (head only) | 50 | ~37s | ~30 min |
| Stage 2 (conv5 + head) | 50 | ~40s | ~33 min |
| Test set evaluation | — | — | ~10s |
| Postgres persistence (1,600 rows bulk) | — | — | ~2s |
| **Per fold total** | | | **~65 min** |

Five folds: **~5h25**, comfortably within Kaggle's 9-hour session limit. Phase 1 single-fold sanity check was validated at exactly 65 minutes (see [Phase 1 results](../phases/phase-1-vgg16-baseline.md#results)).
