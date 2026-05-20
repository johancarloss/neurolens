"""NeuroLens — Hello World kernel.

Validates the dual-write tracking pipeline end-to-end on Kaggle by:

1. Reading DATABASE_URL and WANDB_API_KEY from Kaggle Secrets
2. Cloning the public neurolens repo into the kernel working dir
3. Installing minimal extra dependencies (rest are in the Kaggle image)
4. Asserting the Kaggle-attached dataset has the expected 4-class structure
5. Instantiating CompositeLogger -> writes a single epoch to W&B + Postgres + JSONL
6. Printing references (W&B URL, Postgres run_id, JSONL path) for verification

Success criteria (verified manually after run):
- W&B shows a new run in group "hello-world"
- Postgres `neurolens.runs` has a new row with status='completed'
- /kaggle/working/jsonl/<run_id>.jsonl contains 3 events (run_start, epoch_end, run_complete)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

# ============================================================================
# 1. Load secrets from Kaggle Secrets API
# ============================================================================
us = UserSecretsClient()
os.environ["DATABASE_URL"] = us.get_secret("DATABASE_URL")
os.environ["WANDB_API_KEY"] = us.get_secret("WANDB_API_KEY")

# Sanity (do NOT print secret values)
assert os.environ["DATABASE_URL"].startswith("postgresql://"), (
    "DATABASE_URL must be a postgresql:// connection string"
)
assert len(os.environ["WANDB_API_KEY"]) >= 20, "WANDB_API_KEY looks too short"

# ============================================================================
# 2. Clone the neurolens repo (public)
# ============================================================================
REPO_DIR = Path("/kaggle/working/neurolens-repo")
if not REPO_DIR.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/johancarloss/neurolens.git",
            str(REPO_DIR),
        ],
        check=True,
    )

# ============================================================================
# 3. Install minimal extras (Kaggle has torch, pandas, numpy etc. preinstalled)
# ============================================================================
subprocess.run(
    ["pip", "install", "-q", "psycopg2-binary>=2.9", "wandb>=0.18", "tenacity>=8.0"],
    check=True,
)

# Make neurolens package importable
sys.path.insert(0, str(REPO_DIR / "src"))

from neurolens.tracking.composite import CompositeLogger  # noqa: E402

# ============================================================================
# 4. Validate dataset attached at /kaggle/input/
# ----------------------------------------------------------------------------
# Use recursive glob to FIND the class folders wherever they are mounted.
# Kaggle's path conventions change; we don't assume a fixed slug or layout.
# Strategy: find any directory named "glioma" in /kaggle/input; its parent
# is the train (or root) folder. Walk up one more level to get DATA_ROOT.
# ============================================================================
EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
INPUT_ROOT = Path("/kaggle/input")


def print_tree(root: Path, max_depth: int = 3, _depth: int = 0) -> None:
    """Print a directory tree up to `max_depth` for diagnostics."""
    if _depth > max_depth:
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


print("=" * 60)
print(f"Tree under {INPUT_ROOT} (depth ≤ 3):")
print_tree(INPUT_ROOT, max_depth=3)
print("=" * 60)

# Recursive search: find all directories named like our expected classes
glioma_dirs = list(INPUT_ROOT.rglob("glioma"))
glioma_dirs = [p for p in glioma_dirs if p.is_dir()]

assert glioma_dirs, (
    f"Could not find any 'glioma' folder under {INPUT_ROOT}. "
    f"Tree shown above — verify the dataset is actually attached and contains "
    f"the expected 4 classes."
)


# Pick the deepest "glioma" that has siblings matching the other 3 classes
def _is_valid_class_parent(parent: Path) -> bool:
    """Returns True if `parent` contains all 4 expected classes as subfolders."""
    return all((parent / c).is_dir() for c in EXPECTED_CLASSES)


class_parents = sorted(
    {g.parent for g in glioma_dirs if _is_valid_class_parent(g.parent)},
    key=lambda p: str(p),
)

assert class_parents, (
    f"Found 'glioma' folder(s) but none of their parents contain all 4 classes "
    f"({EXPECTED_CLASSES}). Found glioma at: {glioma_dirs}"
)

# Detect Training/Testing layout (2 class_parents) vs flat (1 class_parent)
if len(class_parents) == 2:
    # Pick the one whose name starts with 'Train' as training, the other as test
    training_dir = next(
        (p for p in class_parents if p.name.lower().startswith("train")),
        class_parents[0],
    )
    testing_dir = next((p for p in class_parents if p != training_dir), None)
    DATA_ROOT = training_dir.parent
    structure = f"Training/Testing layout (root={DATA_ROOT})"
elif len(class_parents) == 1:
    training_dir = class_parents[0]
    testing_dir = None
    DATA_ROOT = training_dir
    structure = f"flat classes layout (root={DATA_ROOT})"
else:
    # Multiple roots — pick the first as both train/test guard
    training_dir = class_parents[0]
    testing_dir = class_parents[1] if len(class_parents) > 1 else None
    DATA_ROOT = training_dir.parent
    structure = (
        f"multiple class roots found: {class_parents} "
        f"(using first as train, second as test if available)"
    )

print(f"✓ Dataset root resolved to: {DATA_ROOT}")
print(f"✓ Training dir: {training_dir}")
print(f"✓ Testing dir:  {testing_dir}")
print(f"✓ Structure:    {structure}")


def _count_images(folder: Path) -> int:
    """Count common image file extensions inside a folder (non-recursive)."""
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        total += len(list(folder.glob(ext)))
    return total


train_classes = sorted(p.name for p in training_dir.iterdir() if p.is_dir())
assert train_classes == EXPECTED_CLASSES, (
    f"Training classes mismatch. Got {train_classes}, expected {EXPECTED_CLASSES}"
)
train_counts = {c: _count_images(training_dir / c) for c in train_classes}

if testing_dir is not None:
    test_classes = sorted(p.name for p in testing_dir.iterdir() if p.is_dir())
    assert test_classes == EXPECTED_CLASSES, (
        f"Testing classes mismatch. Got {test_classes}, expected {EXPECTED_CLASSES}"
    )
    test_counts = {c: _count_images(testing_dir / c) for c in test_classes}
else:
    test_counts = dict.fromkeys(EXPECTED_CLASSES, 0)

print("=" * 60)
print("Dataset validated:")
print(f"  Training counts: {train_counts}  (total: {sum(train_counts.values())})")
print(f"  Testing counts:  {test_counts}  (total: {sum(test_counts.values())})")
print("=" * 60)

# ============================================================================
# 5. Dual-write via CompositeLogger
# ============================================================================
logger = CompositeLogger(
    project="neurolens",
    experiment="hello-world",
    config={
        "arch": "none",
        "stage": 0,
        "phase": "infrastructure-validation",
        "train_counts": train_counts,
        "test_counts": test_counts,
    },
    jsonl_dir=Path("/kaggle/working/jsonl"),
    wandb_tags=["hello-world", "phase-0", "dual-write-validation"],
    kaggle_kernel_url="https://www.kaggle.com/code/johancarloss/neurolens-hello-world",
)

# Log a mock epoch (proves log() works)
logger.log(
    epoch=0,
    phase="test",  # 'phase' as enforced by schema CHECK constraint
    metrics={
        "total_train_images": sum(train_counts.values()),
        "total_test_images": sum(test_counts.values()),
        "num_classes": len(train_classes),
    },
)

logger.finish(
    status="completed",
    final_metrics={
        "dataset_validated": 1.0,
        "expected_classes_match": 1.0,
    },
)

# ============================================================================
# 6. Print references for manual verification
# ============================================================================
print("=" * 60)
print("DUAL-WRITE VALIDATION SUMMARY")
print("=" * 60)
print(f"  W&B run URL:       {logger.wandb_run.url if logger.wandb_run else 'UNAVAILABLE'}")
print(f"  Postgres run_id:   {logger.run_id}")
print(f"  JSONL path:        {logger.jsonl_path}")
print("=" * 60)
print("Verification commands (run from local VPS):")
print()
print("  # 1. Check Postgres:")
print("  psql -h <vps> -U neurolens_writer -d neurolens \\")
print(f'      -c "SELECT * FROM neurolens.runs WHERE id = {logger.run_id}"')
print()
print("  # 2. Download JSONL:")
print("  kaggle kernels output johancarloss/neurolens-hello-world -p ./kernel-output/hello-world")
print("=" * 60)
