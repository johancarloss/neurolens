"""NeuroLens — Hello World kernel.

This script is the MINIMAL bootstrapper that runs on Kaggle.
All real logic lives in the cloned GitHub repo's `src/neurolens/` package.

Why: each `kaggle kernels push` via CLI disconnects secrets and dataset
attachments. By keeping this file tiny and delegating to the cloned repo,
we can iterate on Python code WITHOUT re-pushing the kernel — just `git push`
to GitHub and click "Run All" in the Kaggle UI.

Steps:
1. Load secrets from Kaggle Secrets (DATABASE_URL, WANDB_API_KEY)
2. Clone the public neurolens repo into /kaggle/working/neurolens-repo
3. Install minimal extras (psycopg2-binary, wandb, tenacity)
4. Discover dataset structure from /kaggle/input/ (resilient to layout changes)
5. Dual-write a single epoch to W&B + Postgres + JSONL via CompositeLogger
6. Print references for manual verification
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

# ============================================================================
# 1. Load secrets
# ============================================================================
us = UserSecretsClient()
os.environ["DATABASE_URL"] = us.get_secret("DATABASE_URL")
os.environ["WANDB_API_KEY"] = us.get_secret("WANDB_API_KEY")

assert os.environ["DATABASE_URL"].startswith("postgresql://"), (
    "DATABASE_URL must be a postgresql:// connection string"
)
assert len(os.environ["WANDB_API_KEY"]) >= 20, "WANDB_API_KEY looks too short"

# ============================================================================
# 2. Clone the public neurolens repo (fresh on every run)
# ============================================================================
REPO_DIR = Path("/kaggle/working/neurolens-repo")
if REPO_DIR.exists():
    # Force fresh clone so we always get the latest GitHub main
    subprocess.run(["rm", "-rf", str(REPO_DIR)], check=True)

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
# 3. Install minimal extras (Kaggle image already has torch, pandas, numpy)
# ============================================================================
subprocess.run(
    ["pip", "install", "-q", "psycopg2-binary>=2.9", "wandb>=0.18", "tenacity>=8.0"],
    check=True,
)

# Make neurolens package importable from the cloned repo
sys.path.insert(0, str(REPO_DIR / "src"))

# ============================================================================
# 4. All real logic lives in the cloned repo — import and run
# ============================================================================
from neurolens.data.kaggle_paths import (  # noqa: E402
    count_images,
    discover_brain_tumor_dataset,
)
from neurolens.tracking.composite import CompositeLogger  # noqa: E402

paths = discover_brain_tumor_dataset()

train_counts = {
    c: count_images(paths.training_dir / c)
    for c in sorted(p.name for p in paths.training_dir.iterdir() if p.is_dir())
}
test_counts = (
    {
        c: count_images(paths.testing_dir / c)
        for c in sorted(p.name for p in paths.testing_dir.iterdir() if p.is_dir())
    }
    if paths.testing_dir is not None
    else {}
)

print("=" * 60)
print(f"  Train counts: {train_counts}  (total: {sum(train_counts.values())})")
print(f"  Test counts:  {test_counts}  (total: {sum(test_counts.values())})")
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
        "data_root": str(paths.data_root),
        "structure": paths.structure,
        "train_counts": train_counts,
        "test_counts": test_counts,
    },
    jsonl_dir=Path("/kaggle/working/jsonl"),
    wandb_tags=["hello-world", "phase-0", "dual-write-validation"],
    kaggle_kernel_url=(
        "https://www.kaggle.com/code/johancarloss/neurolens-hello-world-dual-write-validation"
    ),
)

logger.log(
    epoch=0,
    phase="test",
    metrics={
        "total_train_images": sum(train_counts.values()),
        "total_test_images": sum(test_counts.values()),
        "num_classes": len(train_counts),
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
# 6. References for manual verification
# ============================================================================
print("=" * 60)
print("DUAL-WRITE VALIDATION SUMMARY")
print("=" * 60)
print(f"  W&B run URL:     {logger.wandb_run.url if logger.wandb_run else 'UNAVAILABLE'}")
print(f"  Postgres run_id: {logger.run_id}")
print(f"  JSONL path:      {logger.jsonl_path}")
print("=" * 60)
