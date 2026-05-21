"""NeuroLens Runner — universal entry point for every Kaggle job.

This kernel is pushed to Kaggle exactly ONCE. From then on, every job type
(VGG16 training, ResNet50 training, XAI batch generation, ...) is selected
via the ``configs/active_run.yaml`` file in the repo.

Why a file instead of UI variables: Kaggle Notebook editor does NOT have a
"Variables" UI (confirmed via official docs — only Secrets/Datasets/Models).
Using a versioned YAML in the repo gives us deterministic, auditable
"switch the active run" semantics: edit, ``git push``, click Run All.

Responsibilities of THIS script (no domain logic):
    1. Load credentials from Kaggle Secrets
    2. Clone the public neurolens repo (always fresh -> latest main)
    3. Install minimal extra deps not on the Kaggle base image
    4. Read ``configs/active_run.yaml`` from the cloned repo
    5. Dispatch to the right ``main(config_profile=...)`` based on job_type

All actual logic lives in ``src/neurolens/...`` inside the cloned repo.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import yaml
from kaggle_secrets import UserSecretsClient

# ============================================================================
# JOB registry — maps job_type -> dotted import path with a ``main(config_profile)``
# Add new jobs in future phases by appending one line.
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

# Make neurolens importable and run from the repo root.
sys.path.insert(0, str(REPO_DIR / "src"))
os.chdir(REPO_DIR)

# ============================================================================
# 4. Read configs/active_run.yaml from the cloned repo
# ============================================================================
ACTIVE_RUN_PATH = REPO_DIR / "configs" / "active_run.yaml"
if not ACTIVE_RUN_PATH.exists():
    raise FileNotFoundError(
        f"Missing {ACTIVE_RUN_PATH}. The runner reads this file to decide what "
        f"to execute. Create it in the repo and `git push`."
    )

with ACTIVE_RUN_PATH.open() as f:
    active_run = yaml.safe_load(f) or {}

job_type = active_run.get("job_type", "train_vgg16")
config_profile = active_run.get("config_profile", "vgg16")

print(f"[runner] active_run.yaml -> job_type={job_type!r} config_profile={config_profile!r}")

# ============================================================================
# 5. Dispatch
# ============================================================================
if job_type not in JOB_TYPES:
    raise ValueError(
        f"Unknown job_type: {job_type!r}. Supported: {sorted(JOB_TYPES)}. "
        f"Edit configs/active_run.yaml in the repo and git push."
    )

module = importlib.import_module(JOB_TYPES[job_type])
module.main(config_profile=config_profile)
