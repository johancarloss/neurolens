"""NeuroLens Runner — universal entry point for every Kaggle job.

This kernel is pushed to Kaggle exactly ONCE. From then on, every job type
(VGG16 training, ResNet50 training, XAI batch generation, ...) is selected
via the ``JOB_TYPE`` environment variable, set through the Kaggle UI under
``Add-ons -> Variables``.

Why: ``kaggle kernels push`` unlinks secrets and dataset attachments every
time. By having a single universal runner, we attach secrets + dataset
ONCE for the entire project. To switch jobs, only the ``JOB_TYPE``
variable on the UI needs to change.

Responsibilities of THIS script (the only ones — no domain logic):
    1. Load credentials from Kaggle Secrets
    2. Clone the public neurolens repo (always fresh -> latest main)
    3. Install minimal extra deps not on the Kaggle base image
    4. Dispatch to the right ``main()`` based on ``JOB_TYPE``

All actual logic lives in ``src/neurolens/...`` inside the cloned repo.
This file MUST stay short. If it grows past ~80 lines, suspect domain leak.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

# ============================================================================
# JOB registry — maps JOB_TYPE -> dotted import path with a ``main()``
# ----------------------------------------------------------------------------
# Adding a new job in a future phase: append one line below + ensure the
# module exposes a ``main()`` callable. ZERO changes anywhere else needed.
# ============================================================================
JOB_TYPES: dict[str, str] = {
    "train_vgg16": "neurolens.training.run_vgg16",
    # "train_resnet50": "neurolens.training.run_resnet50",  # Phase 2
    # "xai_batch":      "neurolens.xai.run_batch",          # Phase 3
}


# ============================================================================
# 1. Secrets
# ============================================================================
us = UserSecretsClient()
os.environ["DATABASE_URL"] = us.get_secret("DATABASE_URL")
os.environ["WANDB_API_KEY"] = us.get_secret("WANDB_API_KEY")

assert os.environ["DATABASE_URL"].startswith("postgresql://"), (
    "DATABASE_URL must be a postgresql:// connection string"
)
assert len(os.environ["WANDB_API_KEY"]) >= 20, "WANDB_API_KEY looks too short"

# ============================================================================
# 2. Clone fresh repo (always latest main)
# ============================================================================
REPO_DIR = Path("/kaggle/working/neurolens-repo")
if REPO_DIR.exists():
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
# 3. Install extras (Kaggle image already has torch, sklearn, etc.)
# ============================================================================
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "psycopg2-binary>=2.9",
        "wandb>=0.18",
        "tenacity>=8.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
    ],
    check=True,
)

# Make neurolens importable and run from the repo root so relative config
# paths inside run_*.py modules resolve correctly.
sys.path.insert(0, str(REPO_DIR / "src"))
os.chdir(REPO_DIR)

# ============================================================================
# 4. Dispatch
# ============================================================================
JOB_TYPE = os.environ.get("JOB_TYPE", "train_vgg16")
print(f"[runner] JOB_TYPE={JOB_TYPE}")

if JOB_TYPE not in JOB_TYPES:
    raise ValueError(
        f"Unknown JOB_TYPE: {JOB_TYPE!r}. "
        f"Supported: {sorted(JOB_TYPES)}. "
        f"Set JOB_TYPE in Add-ons -> Variables on the Kaggle UI."
    )

module = importlib.import_module(JOB_TYPES[JOB_TYPE])
module.main()
