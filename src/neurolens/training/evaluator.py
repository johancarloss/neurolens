"""Test-set evaluator returning per-class metrics + confusion matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> dict[str, Any]:
    """Run inference over ``loader`` and compute classification metrics.

    Args:
        model: trained model (will be set to eval mode).
        loader: test-set DataLoader yielding ``(image_tensor, label_int)``.
        device: CUDA or CPU device for the forward pass.
        classes: ordered class names used to label the report.

    Returns:
        Dict with:
            - ``overall_accuracy`` (float)
            - ``macro_f1``, ``weighted_f1`` (float)
            - ``per_class`` (dict[str, dict[str, float]]) — precision/recall/f1
            - ``confusion_matrix`` (list[list[int]]) — JSON-serializable
            - ``preds`` (np.ndarray) — predicted class indices
            - ``targets`` (np.ndarray) — true class indices
            - ``probs`` (np.ndarray) — softmax probabilities, shape (N, num_classes)
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(probs.argmax(1).cpu().numpy())
        all_targets.append(y.numpy())
        all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    report = classification_report(
        targets, preds, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(targets, preds)

    return {
        "overall_accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {cls: report[cls] for cls in classes},
        "confusion_matrix": cm.tolist(),
        "preds": preds,
        "targets": targets,
        "probs": probs,
    }
