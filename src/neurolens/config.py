"""Pydantic models for NeuroLens training configs.

Configs are loaded from YAML and validated at import time so that
typos in hyperparameters fail fast (well before the GPU is allocated).

Reference: docs/private/blueprint/03-stack-and-conventions.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AugmentationConfig(BaseModel):
    """Train-time augmentation parameters (matches Wong et al. 2025)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shear_range: float = Field(default=0.2, ge=0.0, le=1.0)
    zoom_range: float = Field(default=0.2, ge=0.0, le=1.0)
    horizontal_flip: bool = True


class TrainConfig(BaseModel):
    """Top-level training configuration.

    Validation runs at construction time. Common mistakes (typos, wrong
    LR magnitude, unsupported optimizer) raise here rather than 30 minutes
    into training.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Model identity ----------------------------------------------------
    arch: Literal["vgg16", "resnet50"]
    stage: Literal[1, 2]

    # --- Optimization ------------------------------------------------------
    lr: float = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=50, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    loss: Literal["cross_entropy"] = "cross_entropy"

    # --- Data / augmentation ----------------------------------------------
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    image_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4, ge=0)

    # --- Model head --------------------------------------------------------
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)

    # --- Reproducibility ---------------------------------------------------
    seed: int = 42

    # --- Cross-validation --------------------------------------------------
    cv_folds: int = Field(default=5, ge=2, le=10)

    # --- Early stopping (optional) ----------------------------------------
    early_stopping_patience: int | None = Field(default=None, ge=1)
    save_best_only: bool = True

    @field_validator("lr")
    @classmethod
    def _sanity_check_lr(cls, value: float) -> float:
        """Reject obviously wrong learning rates (e.g., 1.0 from a missing scale)."""
        if value > 0.1:
            raise ValueError(f"learning rate {value} is suspiciously large (expected ≤ 0.1)")
        if value < 1e-6:
            raise ValueError(f"learning rate {value} is suspiciously small (expected ≥ 1e-6)")
        return value


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a YAML training config.

    Args:
        path: filesystem path to the YAML file.

    Returns:
        Validated TrainConfig (frozen — safe to share across threads).

    Raises:
        FileNotFoundError: if the YAML does not exist.
        pydantic.ValidationError: if any field fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return TrainConfig(**data)
