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
assert os.environ["DATABASE_URL"].startswith("postgresql://"), \
    "DATABASE_URL must be a postgresql:// connection string"
assert len(os.environ["WANDB_API_KEY"]) >= 20, "WANDB_API_KEY looks too short"

# ============================================================================
# 2. Clone the neurolens repo (public)
# ============================================================================
REPO_DIR = Path("/kaggle/working/neurolens-repo")
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/johancarloss/neurolens.git",
         str(REPO_DIR)],
        check=True,
    )

# ============================================================================
# 3. Install minimal extras (Kaggle has torch, pandas, numpy etc. preinstalled)
# ============================================================================
subprocess.run(
    ["pip", "install", "-q",
     "psycopg2-binary>=2.9",
     "wandb>=0.18",
     "tenacity>=8.0"],
    check=True,
)

# Make neurolens package importable
sys.path.insert(0, str(REPO_DIR / "src"))

from neurolens.tracking.composite import CompositeLogger  # noqa: E402

# ============================================================================
# 4. Validate dataset attached at /kaggle/input/brain-tumor-mri-dataset/
# ============================================================================
DATA_ROOT = Path("/kaggle/input/brain-tumor-mri-dataset")
training_dir = DATA_ROOT / "Training"
testing_dir = DATA_ROOT / "Testing"

assert training_dir.exists(), f"Missing dataset folder: {training_dir}"
assert testing_dir.exists(), f"Missing dataset folder: {testing_dir}"

EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

train_classes = sorted(p.name for p in training_dir.iterdir() if p.is_dir())
test_classes = sorted(p.name for p in testing_dir.iterdir() if p.is_dir())

assert train_classes == EXPECTED_CLASSES, (
    f"Training classes mismatch. Got {train_classes}, expected {EXPECTED_CLASSES}"
)
assert test_classes == EXPECTED_CLASSES, (
    f"Testing classes mismatch. Got {test_classes}, expected {EXPECTED_CLASSES}"
)

train_counts = {c: len(list((training_dir / c).glob("*.jpg"))) for c in train_classes}
test_counts = {c: len(list((testing_dir / c).glob("*.jpg"))) for c in test_classes}

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
print(f"  psql -h <vps> -U neurolens_writer -d neurolens \\")
print(f"      -c \"SELECT * FROM neurolens.runs WHERE id = {logger.run_id}\"")
print()
print("  # 2. Download JSONL:")
print("  kaggle kernels output johancarloss/neurolens-hello-world -p ./kernel-output/hello-world")
print("=" * 60)
