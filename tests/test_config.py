"""Tests for src/neurolens/config.py and YAML configs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurolens.config import (
    AugmentationConfig,
    TrainConfig,
    XaiConfig,
    load_config,
    load_xai_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_vgg16_stage1_yaml_loads() -> None:
    """The committed vgg16_stage1.yaml must load cleanly with the right values."""
    cfg = load_config(CONFIGS_DIR / "vgg16_stage1.yaml")
    assert cfg.arch == "vgg16"
    assert cfg.stage == 1
    assert cfg.lr == 0.001
    assert cfg.epochs == 50
    assert cfg.batch_size == 32
    assert cfg.augmentation.shear_range == 0.2
    assert cfg.augmentation.zoom_range == 0.2
    assert cfg.augmentation.horizontal_flip is True
    assert cfg.dropout == 0.5
    assert cfg.cv_folds == 5
    assert cfg.seed == 42


def test_vgg16_stage2_yaml_loads() -> None:
    """Stage 2 YAML must have lr=1e-4 and stage=2; everything else identical."""
    cfg = load_config(CONFIGS_DIR / "vgg16_stage2.yaml")
    assert cfg.arch == "vgg16"
    assert cfg.stage == 2
    assert cfg.lr == 0.0001
    assert cfg.epochs == 50


def test_resnet50_stage1_yaml_loads() -> None:
    """The committed resnet50_stage1.yaml must load with matching VGG16 hyperparams."""
    cfg = load_config(CONFIGS_DIR / "resnet50_stage1.yaml")
    assert cfg.arch == "resnet50"
    assert cfg.stage == 1
    assert cfg.lr == 0.001
    assert cfg.epochs == 50
    assert cfg.seed == 42  # same seed as VGG16 -> identical folds (fair comparison)
    assert cfg.cv_folds == 5
    assert cfg.target_fold is None  # all 5 folds


def test_resnet50_stage2_yaml_loads() -> None:
    """Stage 2 YAML must drop lr 10x and set stage=2."""
    cfg = load_config(CONFIGS_DIR / "resnet50_stage2.yaml")
    assert cfg.arch == "resnet50"
    assert cfg.stage == 2
    assert cfg.lr == 0.0001


def test_resnet50_smoke_yaml_loads_with_caps() -> None:
    """Smoke configs must single-fold and cap samples for a <2 min micro run."""
    cfg = load_config(CONFIGS_DIR / "resnet50_smoke_stage1.yaml")
    assert cfg.arch == "resnet50"
    assert cfg.epochs == 1
    assert cfg.target_fold == 0
    assert cfg.train_samples_per_class == 50
    assert cfg.test_samples_per_class == 20


def test_invalid_arch_rejected() -> None:
    """Pydantic Literal validator must reject unsupported archs."""
    with pytest.raises(ValidationError):
        TrainConfig(arch="alexnet", stage=1, lr=1e-3)  # type: ignore[arg-type]


def test_invalid_stage_rejected() -> None:
    """Stage must be 1 or 2."""
    with pytest.raises(ValidationError):
        TrainConfig(arch="vgg16", stage=3, lr=1e-3)  # type: ignore[arg-type]


def test_suspiciously_large_lr_rejected() -> None:
    """The lr sanity check catches LR=1.0 (likely a forgotten /1000)."""
    with pytest.raises(ValidationError, match="suspiciously large"):
        TrainConfig(arch="vgg16", stage=1, lr=1.0)


def test_suspiciously_small_lr_rejected() -> None:
    """The lr sanity check catches LR=1e-9 (likely a typo)."""
    with pytest.raises(ValidationError, match="suspiciously small"):
        TrainConfig(arch="vgg16", stage=1, lr=1e-9)


def test_extra_fields_rejected() -> None:
    """Pydantic strict mode forbids unknown fields (catches typos in YAML keys)."""
    with pytest.raises(ValidationError):
        TrainConfig(arch="vgg16", stage=1, lr=1e-3, learning_rate=1e-3)  # type: ignore[call-arg]


def test_load_config_missing_file_raises() -> None:
    """Missing file must raise FileNotFoundError (not a cryptic pydantic error)."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/nonexistent/config.yaml")


def test_augmentation_defaults() -> None:
    """AugmentationConfig with no args returns Wong's defaults."""
    aug = AugmentationConfig()
    assert aug.shear_range == 0.2
    assert aug.zoom_range == 0.2
    assert aug.horizontal_flip is True


def test_xai_default_yaml_loads() -> None:
    """The committed xai_default.yaml validates and explains both architectures."""
    cfg = load_xai_config(CONFIGS_DIR / "xai_default.yaml")
    assert [a.name for a in cfg.archs] == ["vgg16", "resnet50"]
    # selection draws from the explained folds (so reason matches checkpoint)
    arch_run_ids = {a.run_id for a in cfg.archs}
    assert arch_run_ids.issubset(set(cfg.selection_run_ids))
    assert cfg.num_images > 0


def test_xai_smoke_yaml_loads() -> None:
    """The smoke profile validates and is a tiny single-arch run."""
    cfg = load_xai_config(CONFIGS_DIR / "xai_smoke.yaml")
    assert len(cfg.archs) == 1
    assert cfg.num_images <= 5
    assert cfg.lime_stability_runs == 1


def test_xai_invalid_arch_rejected() -> None:
    """XaiArchSpec arch name is a Literal — unknown archs fail."""
    with pytest.raises(ValidationError):
        XaiConfig(
            archs=[{"name": "alexnet", "checkpoint": "x.pt", "run_id": 1}],  # type: ignore[list-item]
            selection_run_ids=[1],
            num_images=10,
            output_dir="/tmp/xai",
        )


def test_xai_requires_at_least_one_arch() -> None:
    """An empty archs list is rejected (nothing to explain)."""
    with pytest.raises(ValidationError):
        XaiConfig(archs=[], selection_run_ids=[1], num_images=10, output_dir="/tmp/xai")
