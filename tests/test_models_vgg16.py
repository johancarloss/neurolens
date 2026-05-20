"""Tests for src/neurolens/models/vgg16.py and factory.py."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neurolens.models.factory import build_model, get_target_layer_for_gradcam
from neurolens.models.vgg16 import build_vgg16, unfreeze_conv5


def test_vgg16_stage1_forward_shape() -> None:
    """A stage-1 VGG16 forwards (B, 3, 224, 224) -> (B, 4)."""
    model = build_vgg16(num_classes=4, stage=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_vgg16_stage1_features_frozen() -> None:
    """Stage 1: all ``features`` params frozen; classifier params trainable."""
    model = build_vgg16(stage=1)
    assert all(not p.requires_grad for p in model.features.parameters()), (
        "features must be frozen in stage 1"
    )
    assert any(p.requires_grad for p in model.classifier.parameters()), (
        "classifier must be trainable in stage 1"
    )


def test_vgg16_stage2_conv5_unfrozen_early_features_still_frozen() -> None:
    """Stage 2: conv5 trainable; conv1..conv4 still frozen."""
    model = build_vgg16(stage=2)
    # conv5 (idx >= 24) trainable
    assert any(
        p.requires_grad
        for idx in range(24, len(model.features))
        for p in model.features[idx].parameters()
    ), "conv5 must be trainable in stage 2"
    # everything before idx 24 still frozen
    assert all(
        not p.requires_grad for idx in range(0, 24) for p in model.features[idx].parameters()
    ), "early features must remain frozen in stage 2"


def test_unfreeze_conv5_helper_works_inplace() -> None:
    """``unfreeze_conv5`` mutates a stage-1 model to behave like stage 2."""
    model = build_vgg16(stage=1)
    unfreeze_conv5(model)
    assert any(
        p.requires_grad
        for idx in range(24, len(model.features))
        for p in model.features[idx].parameters()
    )


def test_invalid_stage_rejected() -> None:
    """Stages outside {1, 2} must raise."""
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        build_vgg16(stage=3)


def test_factory_dispatches_vgg16() -> None:
    """build_model('vgg16', ...) returns a VGG16."""
    model = build_model("vgg16", num_classes=4, stage=1)
    assert isinstance(model, nn.Module)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 4)


def test_factory_unknown_arch_rejected() -> None:
    """Unknown arch must surface a helpful error."""
    with pytest.raises(ValueError, match="Unknown arch"):
        build_model("alexnet")


def test_gradcam_target_is_last_maxpool() -> None:
    """Grad-CAM target layer for VGG16 is the last MaxPool."""
    model = build_model("vgg16")
    target = get_target_layer_for_gradcam(model, "vgg16")
    assert isinstance(target, nn.MaxPool2d)


def test_gradcam_target_unknown_arch_rejected() -> None:
    """Asking for Grad-CAM target on an unsupported arch must raise."""
    model = build_model("vgg16")
    with pytest.raises(ValueError, match="not defined for arch"):
        get_target_layer_for_gradcam(model, "alexnet")
