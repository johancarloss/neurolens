"""Tests for src/neurolens/data/dataset.py and transforms.py."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from PIL import Image

from neurolens.data.dataset import CLASSES, NUM_CLASSES, build_dataset
from neurolens.data.transforms import eval_transforms, train_transforms


@pytest.fixture
def mock_dataset_root() -> Iterator[Path]:
    """Create a tiny dataset (1 image per class per split) for fast tests."""
    with tempfile.TemporaryDirectory(prefix="neurolens-mock-") as tmp:
        root = Path(tmp)
        for split in ("Training", "Testing"):
            for cls in CLASSES:
                d = root / split / cls
                d.mkdir(parents=True)
                Image.new("RGB", (64, 64)).save(d / "img.jpg")
        yield root


def test_classes_constant_has_four_entries() -> None:
    """NUM_CLASSES must match the CLASSES list length."""
    assert NUM_CLASSES == 4
    assert len(CLASSES) == 4


def test_train_split_loads_with_correct_classes(mock_dataset_root: Path) -> None:
    """build_dataset(split='train') reads Training/ with the 4 expected classes."""
    ds = build_dataset(mock_dataset_root, transform=eval_transforms(), split="train")
    assert ds.classes == CLASSES
    assert len(ds) == 4  # 1 image per class


def test_test_split_loads(mock_dataset_root: Path) -> None:
    """build_dataset(split='test') reads Testing/."""
    ds = build_dataset(mock_dataset_root, transform=eval_transforms(), split="test")
    assert ds.classes == CLASSES
    assert len(ds) == 4


def test_train_transforms_output_shape_and_dtype(mock_dataset_root: Path) -> None:
    """train_transforms returns a (3, 224, 224) float32 tensor."""
    ds = build_dataset(mock_dataset_root, transform=train_transforms(), split="train")
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32
    assert isinstance(y, int)
    assert 0 <= y < NUM_CLASSES


def test_eval_transforms_is_deterministic(mock_dataset_root: Path) -> None:
    """Same image through eval_transforms() twice -> identical tensors."""
    ds = build_dataset(mock_dataset_root, transform=eval_transforms(), split="test")
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert torch.equal(x1, x2)


def test_invalid_split_rejected(mock_dataset_root: Path) -> None:
    """Anything other than 'train' or 'test' must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid split"):
        build_dataset(mock_dataset_root, transform=eval_transforms(), split="validation")


def test_missing_split_folder_raises(tmp_path: Path) -> None:
    """If Training/ doesn't exist, build_dataset raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        build_dataset(tmp_path, transform=eval_transforms(), split="train")


def test_wrong_classes_rejected(tmp_path: Path) -> None:
    """Mismatched class folders must be rejected."""
    bad = tmp_path / "Training"
    for cls in ("apple", "banana", "cherry", "date"):  # wrong names
        (bad / cls).mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(bad / cls / "img.jpg")
    with pytest.raises(ValueError, match="Unexpected class layout"):
        build_dataset(tmp_path, transform=eval_transforms(), split="train")
