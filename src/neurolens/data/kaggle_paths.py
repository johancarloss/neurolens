"""Discover dataset paths on Kaggle Kernels.

Kaggle's mount conventions change over time and across kernel types.
This module provides resilient discovery utilities so the rest of the
package doesn't depend on a fixed path under /kaggle/input/.

Public functions:
- `discover_brain_tumor_dataset()` — return DATA_ROOT, training_dir, testing_dir
- `print_tree(root, depth)` — diagnostic tree dump
- `count_images(folder)` — count common image extensions in a folder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BRAIN_TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_KAGGLE_INPUT_ROOT = Path("/kaggle/input")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


@dataclass(frozen=True)
class BrainTumorPaths:
    """Resolved paths for the brain-tumor MRI dataset on Kaggle."""

    data_root: Path
    training_dir: Path
    testing_dir: Path | None
    structure: str  # human-readable structure description


def print_tree(root: Path, max_depth: int = 3, _depth: int = 0) -> None:
    """Print a directory tree up to `max_depth`. Safe to call on any path."""
    if _depth > max_depth or not root.exists():
        return
    indent = "  " * _depth
    try:
        entries = sorted(root.iterdir())
    except (PermissionError, OSError) as exc:
        print(f"{indent}<unreadable: {exc}>")
        return
    for entry in entries[:50]:
        suffix = "/" if entry.is_dir() else ""
        print(f"{indent}{entry.name}{suffix}")
        if entry.is_dir():
            print_tree(entry, max_depth, _depth + 1)


def count_images(folder: Path) -> int:
    """Count common image-file extensions inside a folder (non-recursive)."""
    if not folder.exists():
        return 0
    return sum(len(list(folder.glob(ext))) for ext in IMAGE_EXTENSIONS)


def _is_valid_class_parent(parent: Path, classes: list[str]) -> bool:
    """Return True if `parent` contains all given class names as subfolders."""
    return all((parent / c).is_dir() for c in classes)


def discover_brain_tumor_dataset(
    input_root: Path = DEFAULT_KAGGLE_INPUT_ROOT,
    classes: list[str] | None = None,
    *,
    verbose: bool = True,
) -> BrainTumorPaths:
    """Find the brain-tumor dataset under `input_root` regardless of layout.

    Strategy:
    1. Recursively search for directories named after the first expected class.
    2. Keep only those whose parent contains ALL 4 expected classes.
    3. Disambiguate: 2 valid parents -> Training/Testing layout;
       1 valid parent -> flat layout (no train/test split).

    Args:
        input_root: where Kaggle mounts datasets. Defaults to /kaggle/input.
        classes: list of expected class names. Defaults to brain-tumor 4 classes.
        verbose: if True, prints the input_root tree and discovery decisions.

    Returns:
        BrainTumorPaths with data_root, training_dir, optional testing_dir, structure.

    Raises:
        AssertionError if no valid dataset layout is found.
    """
    classes = classes if classes is not None else BRAIN_TUMOR_CLASSES

    if verbose:
        print("=" * 60)
        print(f"Tree under {input_root} (depth ≤ 3):")
        print_tree(input_root, max_depth=3)
        print("=" * 60)

    first_class = classes[0]
    candidates = [p for p in input_root.rglob(first_class) if p.is_dir()]
    if not candidates:
        raise AssertionError(
            f"Could not find any '{first_class}' folder under {input_root}. "
            f"Tree above shows actual layout. "
            f"Confirm the dataset is attached to this Kaggle notebook."
        )

    class_parents = sorted(
        {c.parent for c in candidates if _is_valid_class_parent(c.parent, classes)},
        key=lambda p: str(p),
    )

    if not class_parents:
        raise AssertionError(
            f"Found '{first_class}' folder(s) but none of their parents contain "
            f"all expected classes {classes}. Candidates: {candidates}"
        )

    if len(class_parents) == 1:
        training_dir = class_parents[0]
        testing_dir = None
        data_root = training_dir
        structure = "flat classes layout (no Training/Testing split)"
    else:
        # 2+ parents — pick "Train*" as training, the other as testing
        training_dir = next(
            (p for p in class_parents if p.name.lower().startswith("train")),
            class_parents[0],
        )
        testing_dir = next(
            (p for p in class_parents if p != training_dir),
            None,
        )
        data_root = training_dir.parent
        structure = (
            f"Training/Testing layout under {data_root.name}/ "
            f"(extra parents ignored: {class_parents[2:]})"
            if len(class_parents) > 2
            else f"Training/Testing layout under {data_root.name}/"
        )

    if verbose:
        print(f"✓ Dataset root resolved to: {data_root}")
        print(f"✓ Training dir: {training_dir}")
        print(f"✓ Testing dir:  {testing_dir}")
        print(f"✓ Structure:    {structure}")

    return BrainTumorPaths(
        data_root=data_root,
        training_dir=training_dir,
        testing_dir=testing_dir,
        structure=structure,
    )
