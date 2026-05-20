"""NeuroLens — Train VGG16 (Kaggle bootstrap).

MINIMAL bootstrap script. All real logic lives in the cloned GitHub repo's
``src/neurolens/`` package — that way iterating on code only requires
``git push`` + clicking ``Run All`` in the Kaggle UI (no CLI re-push, which
would unlink secrets and dataset).

Pipeline:
1. Load secrets (DATABASE_URL + WANDB_API_KEY) from Kaggle Secrets API
2. Fresh clone of github.com/johancarloss/neurolens (always pulls latest main)
3. Install minimal extras (Kaggle image already has torch/numpy/pandas/sklearn)
4. Delegate to ``neurolens.training.run_vgg16.main()``

Which fold to train is controlled by env var ``FOLD_IDX`` (0..4).
Set it via Add-ons → Variables in the Kaggle UI between runs.
Unset = run all 5 folds sequentially (risky: may exceed 9h Kaggle limit).
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
# 2. Clone the neurolens repo (fresh on every run -> always latest main)
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
# 3. Install minimal extras (Kaggle image already has torch, sklearn, etc.)
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

# Make the cloned neurolens package importable
sys.path.insert(0, str(REPO_DIR / "src"))

# ============================================================================
# 4. Delegate to the entry point in the cloned repo
# ============================================================================
# Change CWD so relative config paths in run_vgg16.py resolve correctly
os.chdir(REPO_DIR)

from neurolens.training.run_vgg16 import main  # noqa: E402

main()
