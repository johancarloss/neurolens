"""Quantitative comparison metrics for the three XAI techniques (Phase 3).

Each XAI technique (Grad-CAM, LIME, SHAP) produces a saliency map over the
224x224 input. To compare them objectively (not just visually), we reduce each
pair of maps to numbers. This module implements the building blocks for the
5 metric families defined in the grill-me principle #7:

1. Agreement  — IoU between technique pairs (do they highlight the same region?)
2. Stability  — variance across repeated LIME runs (is LIME reproducible?)
3. Sparsity   — fraction of the map flagged as important (focused vs diffuse?)
4. Time       — generation cost (passed in by the caller)
5. Per-class  — aggregated downstream in the analysis notebook, from these values

Maps are float arrays in [0, 1]; LIME returns a boolean mask directly.
"""

from __future__ import annotations

import numpy as np


def binarize(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize a heatmap relative to its own maximum.

    A pixel is "important" if it is at least ``threshold`` of the map's peak.
    Using the relative max (not an absolute cutoff) keeps the metric comparable
    across techniques whose raw magnitudes differ.
    """
    peak = heatmap.max()
    if peak <= 0:
        return np.zeros_like(heatmap, dtype=bool)
    return heatmap >= (threshold * peak)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union between two boolean masks.

    1.0 = identical regions, 0.0 = disjoint. The agreement metric.
    """
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def sparsity(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of pixels flagged important (lower = more focused explanation)."""
    binary = binarize(heatmap, threshold)
    return float(binary.sum() / binary.size)


def lime_stability(lime_masks: list[np.ndarray]) -> float:
    """Instability of LIME across repeated runs (lower = more reproducible).

    LIME is stochastic (it samples perturbations), so the same image can yield
    different masks on different runs. We quantify that by the mean per-pixel
    standard deviation across N runs. Returns 0.0 if fewer than 2 runs.
    """
    if len(lime_masks) < 2:
        return 0.0
    stack = np.stack(lime_masks).astype(float)  # (N, H, W)
    return float(stack.std(axis=0).mean())


def compute_all_metrics(
    gradcam_map: np.ndarray,
    lime_mask: np.ndarray,
    shap_map: np.ndarray,
    lime_masks_for_stability: list[np.ndarray] | None = None,
    times: dict[str, float] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute every comparison metric for one prediction's three saliency maps.

    Args:
        gradcam_map: (H, W) float [0, 1] Grad-CAM saliency.
        lime_mask: (H, W) boolean LIME mask.
        shap_map: (H, W) float [0, 1] SHAP saliency.
        lime_masks_for_stability: repeated LIME masks for the stability metric
            (typically 5 runs); None or a single run yields stability 0.0.
        times: generation times in ms, e.g. ``{"time_ms_gradcam": ...}``.
        threshold: relative cutoff for binarizing the float maps.

    Returns:
        Flat dict with IoU (3 pairs), sparsity (3 techniques), LIME stability,
        run count, and whatever timing keys the caller supplied.
    """
    gradcam_bin = binarize(gradcam_map, threshold)
    lime_bin = lime_mask.astype(bool)
    shap_bin = binarize(shap_map, threshold)

    return {
        "iou_gradcam_lime": iou(gradcam_bin, lime_bin),
        "iou_gradcam_shap": iou(gradcam_bin, shap_bin),
        "iou_lime_shap": iou(lime_bin, shap_bin),
        "sparsity_gradcam": sparsity(gradcam_map, threshold),
        "sparsity_lime": float(lime_bin.sum() / lime_bin.size),
        "sparsity_shap": sparsity(shap_map, threshold),
        "lime_stability_std": lime_stability(lime_masks_for_stability or []),
        "lime_num_runs": float(len(lime_masks_for_stability or [])),
        **(times or {}),
    }
