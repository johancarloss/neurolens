"""Tests for the architecture dispatcher in src/neurolens/models/factory.py.

VGG16-specific dispatch is covered in test_models_vgg16.py; this file focuses
on multi-arch behavior added in Phase 2 (ResNet50 + unfreeze_for_stage2).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neurolens.models.factory import (
    build_model,
    get_target_layer_for_gradcam,
    unfreeze_for_stage2,
)


def test_factory_dispatches_resnet50() -> None:
    """build_model('resnet50', ...) returns a working ResNet50."""
    model = build_model("resnet50", num_classes=4, stage=1)
    assert isinstance(model, nn.Module)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 4)


def test_unfreeze_for_stage2_resnet50_dispatches() -> None:
    """unfreeze_for_stage2 unfreezes layer4 for a stage-1 ResNet50."""
    model = build_model("resnet50", stage=1)
    assert all(not p.requires_grad for p in model.layer4.parameters())
    unfreeze_for_stage2(model, "resnet50")
    assert any(p.requires_grad for p in model.layer4.parameters())


def test_unfreeze_for_stage2_vgg16_dispatches() -> None:
    """unfreeze_for_stage2 unfreezes conv5 for a stage-1 VGG16."""
    model = build_model("vgg16", stage=1)
    unfreeze_for_stage2(model, "vgg16")
    assert any(
        p.requires_grad
        for idx in range(24, len(model.features))
        for p in model.features[idx].parameters()
    )


def test_unfreeze_for_stage2_unknown_arch_rejected() -> None:
    """Asking to unfreeze an unsupported arch must raise."""
    model = build_model("resnet50", stage=1)
    with pytest.raises(ValueError, match="No stage-2 unfreeze defined"):
        unfreeze_for_stage2(model, "alexnet")


def test_gradcam_target_resnet50_is_last_bottleneck() -> None:
    """Grad-CAM target dispatch returns layer4[-1] for ResNet50."""
    model = build_model("resnet50")
    target = get_target_layer_for_gradcam(model, "resnet50")
    assert target is model.layer4[-1]
