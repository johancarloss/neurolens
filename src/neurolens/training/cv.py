"""Stratified K-fold cross-validation indices for ImageFolder-style datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_indices(
    targets: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold_idx, train_idx, val_idx)`` for each fold.

    Args:
        targets: per-sample class labels (e.g., ``dataset.targets`` from
            ``torchvision.datasets.ImageFolder``).
        n_splits: number of folds (Rahman et al. 2025 used 5).
        seed: random seed for reproducibility. Same seed -> same splits.

    Yields:
        Tuples of (fold index, training indices, validation indices) where
        the indices are NumPy arrays you can pass to ``torch.utils.data.Subset``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(targets))
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
        yield fold, train_idx, val_idx
