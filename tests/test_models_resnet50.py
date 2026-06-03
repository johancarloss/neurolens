"""Tests for src/neurolens/models/resnet50.py."""

from __future__ import annotations

import pytest
import torch

from neurolens.models.resnet50 import build_resnet50, get_target_layer_for_gradcam, unfreeze_layer4


def test_resnet50_stage1_forward_shape() -> None:
    """A stage-1 ResNet50 forwards (B, 3, 224, 224) -> (B, 4)."""
    model = build_resnet50(num_classes=4, stage=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_resnet50_stage1_backbone_frozen_except_fc() -> None:
    """Stage 1: only the head (model.fc) is trainable; the backbone is frozen."""
    model = build_resnet50(stage=1)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable, "stage 1 must leave the head trainable"
    assert all(name.startswith("fc.") for name in trainable), (
        f"only fc.* should train in stage 1, got: {trainable}"
    )


def test_resnet50_stage2_layer4_unfrozen_earlier_blocks_frozen() -> None:
    """Stage 2: layer4 + fc trainable; layer1..layer3 still frozen."""
    model = build_resnet50(stage=2)
    assert any(p.requires_grad for p in model.layer4.parameters()), (
        "layer4 must be trainable in stage 2"
    )
    for earlier in (model.layer1, model.layer2, model.layer3):
        assert all(not p.requires_grad for p in earlier.parameters()), (
            "early bottleneck blocks must remain frozen in stage 2"
        )


def test_unfreeze_layer4_helper_works_inplace() -> None:
    """``unfreeze_layer4`` mutates a stage-1 model to behave like stage 2."""
    model = build_resnet50(stage=1)
    assert all(not p.requires_grad for p in model.layer4.parameters())
    unfreeze_layer4(model)
    assert any(p.requires_grad for p in model.layer4.parameters())


def test_resnet50_head_has_no_flatten_or_bottleneck() -> None:
    """ResNet50 head is Dropout -> Linear(2048, 4) (no Flatten, no 256-d layer)."""
    model = build_resnet50(num_classes=4)
    linear = model.fc[-1]
    assert isinstance(linear, torch.nn.Linear)
    assert linear.in_features == 2048
    assert linear.out_features == 4


def test_invalid_stage_rejected() -> None:
    """Stages outside {1, 2} must raise."""
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        build_resnet50(stage=3)


def test_gradcam_target_is_last_bottleneck() -> None:
    """Grad-CAM target for ResNet50 is the last bottleneck block (layer4[-1])."""
    model = build_resnet50()
    assert get_target_layer_for_gradcam(model) is model.layer4[-1]
