"""Tests for src/neurolens/training/cv.py."""

from __future__ import annotations

import numpy as np

from neurolens.training.cv import stratified_kfold_indices


def test_5fold_yields_five_splits() -> None:
    """A 5-fold CV must yield exactly 5 (train, val) pairs."""
    targets = sum(([c] * 100 for c in range(4)), [])
    folds = list(stratified_kfold_indices(targets, n_splits=5, seed=42))
    assert len(folds) == 5
    assert [f[0] for f in folds] == [0, 1, 2, 3, 4]


def test_train_and_val_indices_disjoint() -> None:
    """For each fold, train and val sets must be disjoint and union all indices."""
    n = 400
    targets = sum(([c] * (n // 4) for c in range(4)), [])
    all_idx = set(range(n))
    for _, train_idx, val_idx in stratified_kfold_indices(targets, n_splits=5):
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == all_idx


def test_stratification_preserves_class_balance() -> None:
    """Each fold's val set should have roughly equal counts across classes."""
    # 1400 samples per class, matches our real dataset
    targets_array = np.array(sum(([c] * 1400 for c in range(4)), []))
    expected_per_class = 1400 // 5  # 280
    for _, _, val_idx in stratified_kfold_indices(targets_array.tolist(), n_splits=5):
        val_labels = targets_array[val_idx]
        counts = np.bincount(val_labels, minlength=4)
        # StratifiedKFold guarantees |count - expected| <= 1 in each fold
        assert all(abs(int(c) - expected_per_class) <= 1 for c in counts), (
            f"Stratification imbalance: counts={counts.tolist()}"
        )


def test_seed_determinism() -> None:
    """Same seed -> same splits (cross-run reproducibility)."""
    targets = sum(([c] * 100 for c in range(4)), [])
    a = list(stratified_kfold_indices(targets, n_splits=5, seed=42))
    b = list(stratified_kfold_indices(targets, n_splits=5, seed=42))
    for (_, ta, va), (_, tb, vb) in zip(a, b, strict=True):
        assert np.array_equal(ta, tb)
        assert np.array_equal(va, vb)


def test_different_seeds_yield_different_splits() -> None:
    """Different seeds must produce different splits (sanity check)."""
    targets = sum(([c] * 100 for c in range(4)), [])
    a = list(stratified_kfold_indices(targets, n_splits=5, seed=42))
    b = list(stratified_kfold_indices(targets, n_splits=5, seed=7))
    # At least one fold must differ
    any_diff = any(not np.array_equal(av, bv) for (_, _, av), (_, _, bv) in zip(a, b, strict=True))
    assert any_diff
