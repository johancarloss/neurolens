"""Tests for src/neurolens/xai/metrics.py."""

from __future__ import annotations

import numpy as np

from neurolens.xai.metrics import (
    binarize,
    compute_all_metrics,
    iou,
    lime_stability,
    sparsity,
)


def test_iou_identical_masks_is_one() -> None:
    """Two identical masks have perfect overlap."""
    mask = np.array([[True, False], [True, True]])
    assert iou(mask, mask) == 1.0


def test_iou_disjoint_masks_is_zero() -> None:
    """Masks that never overlap have IoU 0."""
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, True], [True, True]])
    assert iou(a, b) == 0.0


def test_iou_empty_masks_is_zero_not_nan() -> None:
    """Two empty masks must return 0.0, not a divide-by-zero NaN."""
    empty = np.zeros((4, 4), dtype=bool)
    assert iou(empty, empty) == 0.0


def test_iou_half_overlap() -> None:
    """Intersection 1, union 3 -> 1/3."""
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    assert iou(a, b) == 1 / 3


def test_binarize_relative_to_peak() -> None:
    """Binarization is relative to the map's own max, not an absolute cutoff."""
    heatmap = np.array([[0.0, 0.4], [0.5, 1.0]])
    # threshold 0.5 -> keep pixels >= 0.5 * 1.0 = 0.5
    result = binarize(heatmap, threshold=0.5)
    assert result.tolist() == [[False, False], [True, True]]


def test_binarize_all_zero_map() -> None:
    """An all-zero map binarizes to all-False (no peak to scale against)."""
    assert not binarize(np.zeros((3, 3))).any()


def test_sparsity_bounds() -> None:
    """Sparsity is 1.0 when everything is above threshold, 0.0 when nothing is."""
    assert sparsity(np.ones((4, 4))) == 1.0
    assert sparsity(np.zeros((4, 4))) == 0.0


def test_lime_stability_identical_runs_is_zero() -> None:
    """Identical LIME runs have zero variance (perfectly stable)."""
    mask = np.array([[True, False], [True, True]])
    assert lime_stability([mask, mask, mask]) == 0.0


def test_lime_stability_single_run_is_zero() -> None:
    """Fewer than 2 runs cannot have variance."""
    assert lime_stability([np.ones((2, 2), dtype=bool)]) == 0.0


def test_lime_stability_detects_variation() -> None:
    """Differing runs produce positive instability."""
    a = np.array([[True, True], [True, True]])
    b = np.array([[False, False], [False, False]])
    assert lime_stability([a, b]) > 0.0


def test_compute_all_metrics_returns_all_keys_and_merges_times() -> None:
    """The aggregator returns every metric and merges caller-supplied times."""
    gradcam = np.array([[0.0, 1.0], [1.0, 0.0]])
    lime_mask = np.array([[False, True], [True, False]])
    shap = np.array([[0.0, 1.0], [1.0, 0.0]])
    times = {"time_ms_gradcam": 50.0, "time_ms_lime": 40000.0, "time_ms_shap": 8000.0}

    result = compute_all_metrics(gradcam, lime_mask, shap, times=times)

    for key in (
        "iou_gradcam_lime",
        "iou_gradcam_shap",
        "iou_lime_shap",
        "sparsity_gradcam",
        "sparsity_lime",
        "sparsity_shap",
        "lime_stability_std",
        "lime_num_runs",
        "time_ms_gradcam",
        "time_ms_lime",
        "time_ms_shap",
    ):
        assert key in result
    # gradcam and shap maps are identical here -> perfect agreement
    assert result["iou_gradcam_shap"] == 1.0
    assert result["time_ms_lime"] == 40000.0
