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
# Auto-discover the dataset folder (slug may differ from what we assume).
# Auto-discover whether classes live under Training/Testing or at the root.
# Tolerant of variations; fails fast with helpful message if structure is off.
# ============================================================================
EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
INPUT_ROOT = Path("/kaggle/input")

print("=" * 60)
print(f"Listing {INPUT_ROOT} contents:")
for entry in INPUT_ROOT.iterdir():
    print(f"  {entry.name}/" if entry.is_dir() else f"  {entry.name}")
print("=" * 60)

# Locate the brain-tumor dataset (try common slugs)
candidate_slugs = [
    "brain-tumor-mri-dataset",
    "brain-tumor-mri",
    "masoudnickparvar-brain-tumor-mri-dataset",
]
DATA_ROOT: Path | None = None
for slug in candidate_slugs:
    candidate = INPUT_ROOT / slug
    if candidate.exists():
        DATA_ROOT = candidate
        break
# Fallback: pick first folder containing expected class names
if DATA_ROOT is None:
    for entry in INPUT_ROOT.iterdir():
        if entry.is_dir() and any(
            (entry / c).exists() or (entry / "Training" / c).exists() for c in EXPECTED_CLASSES
        ):
            DATA_ROOT = entry
            break

assert DATA_ROOT is not None, (
    f"Could not locate brain tumor dataset under {INPUT_ROOT}. Tried slugs: {candidate_slugs}"
)
print(f"✓ Dataset root resolved to: {DATA_ROOT}")

# Detect whether structure is Training/{classes} or {classes} at root
training_dir = DATA_ROOT / "Training"
testing_dir = DATA_ROOT / "Testing"
if training_dir.exists() and testing_dir.exists():
    structure = "Training/Testing layout"
elif all((DATA_ROOT / c).exists() for c in EXPECTED_CLASSES):
    structure = "flat classes at root (no Training/Testing split)"
    # Treat the whole DATA_ROOT as both training_dir and testing_dir = None
    training_dir = DATA_ROOT
    testing_dir = None
else:
    # Print one level deep for diagnostics
    print(f"DATA_ROOT contents: {[p.name for p in DATA_ROOT.iterdir()]}")
    raise AssertionError(
        f"Unrecognized dataset structure under {DATA_ROOT}. "
        f"Expected either Training/{{classes}} + Testing/{{classes}}, "
        f"or {{classes}} directly at root."
    )
print(f"✓ Structure detected: {structure}")

train_classes = sorted(p.name for p in training_dir.iterdir() if p.is_dir())
assert train_classes == EXPECTED_CLASSES, (
    f"Training classes mismatch. Got {train_classes}, expected {EXPECTED_CLASSES}"
)
train_counts = {c: len(list((training_dir / c).glob("*.jpg"))) for c in train_classes}

if testing_dir is not None:
    test_classes = sorted(p.name for p in testing_dir.iterdir() if p.is_dir())
    assert test_classes == EXPECTED_CLASSES, (
        f"Testing classes mismatch. Got {test_classes}, expected {EXPECTED_CLASSES}"
    )
    test_counts = {c: len(list((testing_dir / c).glob("*.jpg"))) for c in test_classes}
else:
    test_counts = dict.fromkeys(EXPECTED_CLASSES, 0)

print("=" * 60)
print("Dataset validated:")
print(f"  Structure:       {structure}")
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
