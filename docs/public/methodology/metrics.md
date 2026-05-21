# Evaluation Metrics

> Reference document — explains every metric reported in the
> [phases/](../phases/) results. Cross-referenced from each phase write-up.

---

## What we measure

Each NeuroLens run produces, on the held-out test set (1,311 images):

1. **Overall accuracy** — global correctness
2. **Per-class precision, recall, F1** — class-level breakdown
3. **Macro F1** — average F1 across classes (equal weight)
4. **Weighted F1** — average F1 weighted by class support
5. **Confusion matrix** — full 4×4 view of classifications and confusions

All five are computed in [`src/neurolens/training/evaluator.py`](../../../src/neurolens/training/evaluator.py) using `sklearn.metrics.classification_report` and `confusion_matrix`. They are emitted to all three sinks of the dual-write tracker.

---

## Accuracy

$$
\text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$

The most intuitive metric. **Value range**: 0 to 1.

**When it fails to be informative**: severely imbalanced datasets. If 90% of the dataset is class A, a degenerate classifier that always predicts A scores 90% accuracy without learning anything. The Brain Tumor MRI dataset is **mildly imbalanced** (1.23:1 max/min ratio), so accuracy is meaningful here — but we always report it alongside macro F1.

---

## Precision, Recall, and the confusion matrix view

For a binary view of one class vs. the rest:

```
                  Predicted: positive  |  Predicted: negative
Actual: positive   True Positive (TP)  |   False Negative (FN)
Actual: negative   False Positive (FP) |   True Negative (TN)
```

$$
\text{precision} = \frac{TP}{TP + FP}, \quad \text{recall} = \frac{TP}{TP + FN}
$$

### Interpretation in medical context

- **Precision** = "of all the times the model said *positive*, how often was it correct?" — measures **false positives**.
- **Recall** (a.k.a. sensitivity) = "of all the actual positives, how many did the model catch?" — measures **false negatives**.

For brain tumor detection:

- **Low recall** = the model misses real tumors. Patient is told they're fine when they're not. **Clinically dangerous.**
- **Low precision** = the model flags tumors that aren't there. Patient gets a scary follow-up biopsy unnecessarily. **Costly and harmful** but less so than missing one.

Most medical screening prioritizes **recall** at the expense of precision. Our classification task does not have an explicit cost asymmetry, so both are reported.

---

## F1 Score

F1 is the **harmonic mean** of precision and recall:

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

The harmonic mean punishes imbalance: a model with precision = 0.99 and recall = 0.20 gets F1 = 0.33, **not** the 0.59 arithmetic average. This is the correct behavior for medical decision-making — you cannot trade recall for precision freely.

### Per-class F1

For multi-class problems, F1 is computed **per class** by binarizing (one-vs-rest):

```
F1(glioma):     treat "glioma" as positive, everything else as negative
F1(meningioma): treat "meningioma" as positive, everything else as negative
F1(notumor):    ...
F1(pituitary):  ...
```

This is the most informative view because it reveals **which class the model struggles with**. In Phase 1 we observed F1(glioma) ≈ 0.90 and F1(pituitary) ≈ 0.99, suggesting the model handles anatomically obvious tumors easily but struggles with morphologically heterogeneous ones — see [Phase 1 analysis](../phases/phase-1-vgg16-baseline.md#per-class-analysis).

### Macro F1

$$
\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^{C} F_1(c)
$$

Average of per-class F1 with **equal weight** for every class. **Rare classes count as much as common classes.** Preferred in medical and imbalanced contexts because it does not hide poor performance on small classes.

### Weighted F1

$$
\text{Weighted } F_1 = \sum_{c=1}^{C} \frac{n_c}{N} \cdot F_1(c)
$$

Average of per-class F1 weighted by the **number of samples** in each class (`n_c` / total `N`). Common classes contribute more. For balanced datasets, weighted F1 ≈ macro F1; for imbalanced datasets, weighted F1 ≈ accuracy.

In the Brain Tumor MRI dataset, the class balance is mild enough that macro F1 and weighted F1 are usually within ~0.005 of each other.

---

## Confusion matrix

A 4 × 4 matrix where rows are **true labels** and columns are **predicted labels**:

```
                  Predicted →
            glioma  meningioma  notumor  pituitary
   glioma    [TP]    [→ men]    [→ no]   [→ pit]
   meningi.  [→ gl]    [TP]     [→ no]   [→ pit]
   notumor   [→ gl]   [→ men]    [TP]    [→ pit]
   pituitary [→ gl]   [→ men]   [→ no]    [TP]
```

The **diagonal** counts correct predictions. **Off-diagonal entries** count specific confusions — e.g., row `glioma`, column `meningioma` is the number of glioma scans the model called meningioma.

The confusion matrix is more informative than any scalar metric: it tells you **which specific class pairs** the model confuses. Phase 3 (XAI) will use this to guide visual analysis of the hardest decision boundaries.

---

## What we DO NOT report (and why)

A few common metrics are intentionally omitted from Phase 1:

- **ROC-AUC**: useful when threshold tuning matters. Our task is fixed multi-class with `argmax`; ROC is conceptually marginal here.
- **Cohen's kappa**: corrects for chance agreement. Useful when classes are extremely imbalanced; ours are not.
- **Top-k accuracy**: relevant when "close enough" is acceptable (ImageNet 1000 classes). For 4 mutually exclusive medical classes, top-1 is what matters.

These may appear in Phase 3 or Phase 5 reports if specific comparisons demand them.

---

## How to read our results

When Phase 1 reports:

```
test_accuracy = 0.9431
macro_f1 = 0.9421
weighted_f1 = 0.9421
per-class F1: glioma 0.8995 | meningioma 0.9288 | notumor 0.9513 | pituitary 0.9889
```

You should read it as:

1. **Macro F1 0.9421** — primary headline. Class-balanced quality measure. Higher is better; 1.0 is perfect.
2. **The four per-class F1s** — diagnostic detail. The 9-point spread between glioma (0.90) and pituitary (0.99) is the actual story.
3. **Accuracy 0.9431** ≈ macro F1 — confirms the class balance is reasonable.
4. **Confusion matrix** (when shown) — the *why* behind the per-class spread.

Reporting a single accuracy number without F1 breakdown would hide where the model actually fails.
