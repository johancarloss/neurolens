"""Tests for src/neurolens/xai/gradcam.py.

Integration-level: builds a real model (ImageNet weights are cached) and runs
Grad-CAM end-to-end on a single synthetic image, on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from neurolens.models.factory import build_model
from neurolens.xai.gradcam import GradCAMExplainer


def _explainer() -> GradCAMExplainer:
    # stage=2 so layer4 is trainable -> gradients reach the Grad-CAM target layer
    model = build_model("resnet50", num_classes=4, stage=2)
    return GradCAMExplainer(model, arch="resnet50", device=torch.device("cpu"))


def test_gradcam_map_shape_and_range() -> None:
    """Grad-CAM returns a (224, 224) map in [0, 1] plus a positive time."""
    explainer = _explainer()
    cam, elapsed_ms = explainer.explain(torch.randn(1, 3, 224, 224), target_class=0)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert elapsed_ms > 0.0


def test_gradcam_defaults_to_argmax_when_no_target() -> None:
    """Passing target_class=None must not crash (uses the model's prediction)."""
    explainer = _explainer()
    cam, _ = explainer.explain(torch.randn(1, 3, 224, 224))
    assert cam.shape == (224, 224)


def test_gradcam_overlay_shape_and_dtype() -> None:
    """Overlay blends a CAM onto an RGB image, returning (H, W, 3) uint8."""
    explainer = _explainer()
    cam, _ = explainer.explain(torch.randn(1, 3, 224, 224), target_class=1)
    rgb = np.random.default_rng(0).random((224, 224, 3)).astype(np.float32)
    overlay = explainer.overlay(rgb, cam)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
