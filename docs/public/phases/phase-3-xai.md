# Phase 3 — Comparative XAI Analysis (Grad-CAM + LIME + SHAP)

> **Status:** ✅ **Phase 3 complete** — three XAI techniques applied to both trained architectures on a targeted image sample.
> **Headline:** the triangulation of Grad-CAM, LIME, and SHAP over 40 focused images × 2 architectures produced **12 findings** — including three that reframe *how* the model errs on glioma, one arguably strong enough to change how the classifier should be trusted clinically.
> **Last updated:** 2026-07-07

---

## Goal

Phase 3 is the scientific core of the project — what differentiates it from Wong et al. (2025), who reported 99.24% accuracy on this dataset **without any XAI**. Accuracy alone tells us *what* the model predicts; XAI tells us *why*, *where it looks*, and *how confident it really is*. Three complementary techniques were applied and compared:

- **Grad-CAM** — gradient-based, "which region of the last convolutional block drove this prediction?"
- **LIME** — perturbation-based (superpixels), "which regions, if hidden, would change the model's mind?"
- **SHAP** — game-theoretic (per-pixel), "what is each pixel's fair contribution to the score?"

The Phase 2 finding (glioma recall stuck at 82.2% independently of architecture — the difficulty is in the data, not the model) gave this phase a **surgical target**: focus the XAI on gliomas, and especially on the `glioma → notumor` false negatives (the clinically most serious error).

---

## What got built

| Module | Purpose |
|--------|---------|
| `src/neurolens/xai/gradcam.py` | Grad-CAM explainer using the factory's arch-aware target layer (VGG16 → `features[-1]`, ResNet50 → `layer4[-1]`) |
| `src/neurolens/xai/lime_explainer.py` | LIME wrapper with SLIC segmentation + grayscale-to-RGB stacking for MRI |
| `src/neurolens/xai/shap_explainer.py` | SHAP `GradientExplainer` (robust to modern PyTorch, unlike DeepExplainer) |
| `src/neurolens/xai/metrics.py` | 5-metric family: IoU (agreement), sparsity (focus), LIME stability (reproducibility), time (cost), plus per-class breakdown downstream |
| `src/neurolens/xai/selection.py` | Stratified image selection from PostgreSQL predictions, prioritizing `glioma → notumor` |
| `src/neurolens/xai/run_batch.py` | Architecture-agnostic orchestrator, invoked as the `xai_batch` job by the universal Kaggle runner |
| `configs/xai_{default,smoke}.yaml` | Production (40 imgs, 2 archs, LIME 1000 samples × 5 stability runs, SHAP 200 samples) and smoke profiles |
| `scripts/plot_xai_case_grids.py` | Reproducible generator for the case grid figures below |
| `requirements-kaggle.txt` | Kernel bootstrap now installs from a versioned file — adding a Kaggle dep is a `git push`, not a kernel re-push |
| `kernel/runner/run.py` | Now registers `train` (arch-agnostic) and `xai_batch` jobs; obsolete `run_vgg16` shim dropped (closes issue #1) |
| `tests/test_xai_*.py` | 30 new unit tests (82 total in the suite) |

---

## Execution

- **Compute:** 76 min on a Kaggle T4 (2026-06-18), full success, no partial failures.
- **Coverage:** 40 test images × 2 architectures × 3 techniques = **240 saliency artifacts + 80 comparison rows** persisted to PostgreSQL. Symmetry (120 artifacts per architecture) is exact — every image completed all three techniques on both models.
- **Sample composition** (stratified selection, identical for both architectures):
  - 20 `glioma → notumor` cases — the dangerous false negative, the primary target
  - 12 `confident_glioma_correct` cases — contrast group ("what the model looks at when it gets it right")
  - 8 `confident_error` cases — other confident misclassifications, for context

**Both architectures explained the same 40 images**, so all cross-architecture comparisons in this write-up are paired case-by-case, not averaged across different samples.

---

## Findings — the numbers

### Finding 1 — The three techniques mostly *disagree* with each other

Pairwise IoU (0 = disjoint, 1 = identical):

| Pair | VGG16 | ResNet50 |
|------|-------|----------|
| Grad-CAM ↔ LIME | 0.113 | 0.114 |
| Grad-CAM ↔ SHAP | 0.003 | 0.001 |
| LIME ↔ SHAP | 0.002 | 0.001 |

Grad-CAM and LIME overlap ~11%; SHAP overlaps ~0% with everything. This is a *feature*, not a bug — the three families are philosophically distinct and were expected to complement, not corroborate. If they agreed 90% they would be redundant. The near-zero SHAP overlap has a mechanical root (Finding 2). What matters methodologically: **triangulation is only useful if the sources are independent**, and here they clearly are.

### Finding 2 — SHAP and Grad-CAM/LIME operate at opposite spatial scales

Sparsity (fraction of pixels flagged as important):

| Technique | VGG16 | ResNet50 |
|-----------|-------|----------|
| Grad-CAM | 12.8% | 8.1% |
| LIME | 8.0% | 8.4% |
| SHAP | **0.1%** | **0.1%** |

Grad-CAM and LIME highlight *regions* (~8–13% of pixels); SHAP highlights *points* (~0.1%). IoU between "8% of area" and "0.1% of points" is near-zero even when both are pointing at the same anatomical location — hence Finding 1.

### Finding 3 — Compute cost: LIME is the bottleneck

Mean generation time per explanation (T4 GPU):

| Technique | VGG16 | ResNet50 |
|-----------|-------|----------|
| Grad-CAM | 21 ms | 17 ms |
| SHAP | 2.96 s | 2.02 s |
| LIME | **11.50 s** | **7.80 s** |

LIME × 5 stability runs ≈ 1 minute per image. This is why the sample was capped at 40 images. ResNet50 is consistently ~30% faster across all three techniques — inherited from its ~5× smaller parameter count (Phase 2, global average pooling).

### Finding 4 — Both models predict *the same class* on every image

**On all 40 images, VGG16 and ResNet50 produced the same prediction** — including all 20 `glioma → notumor` cases (both erred the same glioma the same way). Combined with the paired 82.2% recall from Phase 2, this is now the *second* independent evidence that the glioma difficulty lives in the data, not in the architecture. Two witnesses with radically different designs, telling the same story case by case.

### Finding 5 — They agree on the *class*, but not on *where they look*

Pearson correlation of sparsity between VGG16 and ResNet50 on the same image:

| Technique | Correlation | Reading |
|-----------|-------------|---------|
| **LIME** | **0.66** (strong) | attention is highly consistent across architectures |
| **Grad-CAM** | **0.47** (moderate) | ResNet50 focuses tighter (8%) than VGG16 (13%) |
| **SHAP** | **−0.03** (none) | SHAP is architecture-sensitive |

Methodological consequence: **each technique has a different "architecture sensitivity"**. LIME is the most robust across models because it's black-box (perturbs the image, ignores the internals). Grad-CAM is moderate. SHAP is the most model-specific because it samples internal gradients that vary strongly across architectures. This maps a practical recommendation for future XAI work: **LIME for conclusions that must generalize across models; SHAP for insights specific to one model.**

### Finding 6 — LIME stability: ~5% of pixels vary between runs

Std of per-pixel LIME masks across 5 runs on the same image (lower = more reproducible):

| Arch | Overall | on `confident_error` | on `glioma_correct` | on `glioma → notumor` |
|------|---------|----------------------|---------------------|------------------------|
| VGG16 | 0.045 | 0.055 | **0.037** | 0.046 |
| ResNet50 | 0.055 | **0.035** | 0.056 | **0.063** |

ResNet50's LIME is *most* unstable exactly on the `glioma → notumor` cases (0.063). Even though its Grad-CAM *looks* more focused than VGG16's, LIME reveals that its internal decision oscillates the most on those cases. **ResNet50 hides its uncertainty better than VGG16, but the uncertainty is there.**

### Finding 7 — When the model errs glioma, XAI techniques agree *less* among themselves

IoU (Grad-CAM ↔ LIME) by case type:

| Case | VGG16 | ResNet50 |
|------|-------|----------|
| `confident_error` (other classes) | 0.14 | 0.16 |
| `confident_glioma_correct` | 0.09 | 0.12 |
| `glioma → notumor` | 0.12 | **0.09** |

Techniques align more on confident errors of other classes; they diverge most on glioma misclassifications. **Practical implication**: low inter-technique IoU is an automatable "model is uncertain" detector — without running the model twice, disagreement between XAIs signals fragile decisions.

---

## Findings — the images (what only the original MRI revealed)

The most important findings of the phase came from putting the original MRI **next to** the saliency overlays, not just from the aggregate metrics. Two representative cases were selected as the extremes of the Grad-CAM sparsity distribution among gliomas:

- **Case 1 (error) — `Te-gl_277.jpg`**, VGG16 Grad-CAM sparsity 36.3% (dataset max)
- **Case 2 (correct) — `Te-gl_113.jpg`**, VGG16 Grad-CAM sparsity 5.6% (dataset min)

Both images are gliomas. Case 1 was misread as `notumor` by both models; Case 2 was correctly identified by both, confidently.

### Case 1 — Real glioma read as "notumor"

![Case 1 — real glioma read as notumor](../assets/phase-3-case-error.png)

The tumor is clearly visible in the original MRI (leftmost column) as a **hyperintense white mass in the right frontal lobe** (upper-right of the coronal slice; radiological convention flips left/right). Now look at where the two models attended:

### Finding 8 — When the model errs glioma, it looks at the *wrong region* — this is *inferential* bias, not epistemic uncertainty

- **VGG16 Grad-CAM:** hot spot centered on the **ventricles** (central brain structure), spreading through medial regions. **Does not attend to the frontal lobe.**
- **ResNet50 Grad-CAM:** two peaks, one central and one at the base — **also fails to attend to the frontal lobe.**

The model learned to associate ventricular distortion with tumor presence (a pattern that dominates most training gliomas), so it *always* looks for tumors near the ventricles. A **peripheral glioma that does not distort the ventricles** falls outside the region the model has learned to interrogate. It is not that the model was "spread out because it was unsure" — it was *confidently* looking in the wrong place.

This reframes the earlier "attention spread" observation from a story about model uncertainty into a story about **structural bias**: the model has a learned anatomical decision boundary, and peripheral gliomas are systematically underdetected. This is a substantially stronger clinical finding than the aggregate sparsity numbers alone suggested.

### Finding 9 — SHAP *saw* the tumor even when the model missed it

Look at the rightmost column (SHAP) in Case 1. The bright pixels are concentrated in the **upper-right corner** of both rows — exactly where the tumor is. Not perfectly, but unambiguously. **SHAP is the only one of the three techniques that registered the tumor location**, on both architectures.

Because SHAP measures per-pixel contribution to the output, this means: the tumor signal *did* influence the score, but with insufficient weight to change the argmax. **There is signal in the right place; the model just gave it too little weight relative to the wrong-place signal it learned to trust (Finding 8).**

This flips a natural but wrong story: *"the model failed because there wasn't enough information"*. The information was there. The model failed because it learned to prioritize the wrong information.

### Finding 10 — LIME reveals shortcut learning on correct predictions

![Case 2 — confident correct glioma](../assets/phase-3-case-correct.png)

Case 2 is a **correctly** classified glioma — both models predicted `glioma`, both confident. Success case. Now look at the LIME column:

Both models flag a **thick red band on the dorsal calvarium (skull)** as the important superpixel driving the `glioma` classification, along with regions inside the brain. **Non-cerebral tissue** — skull, skin, acquisition periphery — is contributing to the decision.

This is the textbook signature of **shortcut learning / spurious correlation**: the model has learned features that correlate with the class in *this dataset* but are not the actual disease signal. Possible sources include scanner-specific acquisition patterns (contrast profile, field inhomogeneity), or dataset composition where the glioma scans consistently share acquisition characteristics that differ from the other classes.

**This does not invalidate the ~94% test accuracy from Phases 1 and 2** — the accuracy is real *on this dataset*. But it strongly suggests the accuracy is inflated by dataset-specific features that would not transfer to MRIs from a different scanner or acquisition protocol.

### Finding 11 — A methodological lesson: overlays without the original mislead

The first version of these figures showed only the overlays (no original MRI column). The findings that stand out most in this phase — Findings 8, 9, and 10 — became visible only after adding the original as a reference column. Without the reference, "large red mass" reads as "model uncertain"; with the reference, it reads as "model focused on the ventricles instead of the frontal lobe". Two very different conclusions.

**Rule adopted for future XAI work:** always render the non-annotated reference next to any saliency overlay. Analysis-by-overlay-alone is analysis with half the data.

### Finding 12 — The `glioma → notumor` bottleneck has an actionable diagnosis

Combining Findings 8, 9, and 10:

1. The model has a **structural bias** toward ventricular regions (Finding 8) — peripheral gliomas fall outside its "search area".
2. Signal exists at the correct location, but is **downweighted** relative to the learned shortcut (Finding 9).
3. Even correct predictions rely partly on **non-cerebral features** (Finding 10) — the accuracy is dataset-dependent.

This gives a concrete direction for future improvement: attention-guided training (force the model to attend to actual pathology), region-of-interest masking during training (strip non-brain tissue), and cross-dataset validation on a different scanner (BraTS, TCGA-GBM) to quantify the shortcut effect.

None of this would be visible from the accuracy number alone. This is the *reason* XAI matters — not as decoration on top of a working model, but as a diagnostic tool that reveals *how* it works.

---

## Phase 3 status: complete

- [x] The three techniques implemented and validated (integration tests on CPU with real models)
- [x] 5-metric family computed and persisted for all 80 comparisons (2 archs × 40 images)
- [x] Cross-architecture and error-vs-correct comparisons completed
- [x] Two thesis figures generated with the original MRI as reference column
- [x] 12 findings documented, three of them clinically substantial (Findings 8, 9, 10)
- [x] Results cross-validated against the durable PostgreSQL source (240 artifacts + 80 comparisons)

## Future improvements (parked)

Deliberately deferred so the project can advance to Phase 4 (Gradio demo):

1. **Cross-dataset validation.** Test the models on MRIs from a different source (BraTS, TCGA-GBM) to quantify how much of the ~94% accuracy comes from dataset-specific features vs actual pathology.
2. **Non-brain masking ablation.** Strip skull/periphery from the test set and re-evaluate. If accuracy drops significantly, Finding 10 (shortcut learning) is quantified, not just observed.
3. **Region-of-interest training.** Retrain with attention supervision or brain masks to force the model to attend to parenchyma, and re-run the XAI to see if Finding 8 (ventricular focus) shifts.
4. **More representative cases.** The two thesis figures are the sparsity extremes. Adding 3–5 median-sparsity glioma-error cases would strengthen the generalization of Findings 8 and 10.

---

## Validation and reproducibility

- **Persistence:** every artifact and comparison is stored in PostgreSQL (`xai_artifacts` and `xai_comparisons` tables), linked to the specific prediction row (arch + fold + image). This lets any figure in this write-up be regenerated from the durable source.
- **Case selection is versioned.** The thesis figures were selected as the sparsity extremes among glioma cases via a reproducible SQL query, not hand-picked.
- **To regenerate the figures:**
  ```bash
  uv run kaggle kernels output johancarloss/neurolens-runner -p kernel-output/phase-3-xai/
  uv run python scripts/plot_xai_case_grids.py
  ```
- **To re-run the XAI batch on Kaggle:** point `configs/active_run.yaml` to `xai_default`, push, and Save & Run All.

---

## References

- [Phase 1 — VGG16 Baseline](phase-1-vgg16-baseline.md) — accuracy target and glioma recall bottleneck
- [Phase 2 — Multi-Architecture](phase-2-architectures.md) — establishes that the glioma difficulty is architecture-independent (motivates Phase 3)
- [methodology/model.md](../methodology/model.md) — VGG16 and ResNet50 architectures, and the shared backbone/head split the XAI operates against
- Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
- Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME).
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP).
- Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*. (relevant to Finding 10)
