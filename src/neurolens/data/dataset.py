"""Brain Tumor MRI dataset wrapper (torchvision ImageFolder)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

_SPLIT_DIRNAME = {"train": "Training", "test": "Testing"}


def build_dataset(
    root: str | Path,
    transform: Callable[[Any], Any],
    split: str = "train",
) -> ImageFolder:
    """Build the brain tumor MRI dataset for one split.

    Expected layout under ``root``::

        <root>/
          Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
          Testing/
            <same 4 classes>/

    Args:
        root: dataset root containing ``Training/`` and ``Testing/`` subfolders.
        transform: torchvision transform pipeline (applied to each PIL image).
        split: ``"train"`` -> ``Training/``, ``"test"`` -> ``Testing/``.

    Returns:
        An ``ImageFolder`` whose ``.classes`` are exactly ``CLASSES`` (validated).

    Raises:
        ValueError: if ``split`` is not ``"train"`` or ``"test"``, or if the
            discovered classes don't match the expected 4.
        FileNotFoundError: if the split folder does not exist on disk.
    """
    if split not in _SPLIT_DIRNAME:
        raise ValueError(f"Invalid split '{split}'. Expected one of {list(_SPLIT_DIRNAME)}")

    split_path = Path(root) / _SPLIT_DIRNAME[split]
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_path}")

    dataset = ImageFolder(root=str(split_path), transform=transform)
    if dataset.classes != CLASSES:
        raise ValueError(
            f"Unexpected class layout in {split_path}. Got {dataset.classes}, expected {CLASSES}"
        )
    return dataset
