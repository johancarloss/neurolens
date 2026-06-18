"""Tests for src/neurolens/xai/lime_explainer.py.

Integration-level on CPU. Uses a tiny ``num_samples`` to stay fast — production
uses 1000 (LIME's cost scales with the sample count).
"""

from __future__ import annotations

import numpy as np
import torch

from neurolens.models.factory import build_model
from neurolens.xai.lime_explainer import LimeExplainer


def _explainer() -> LimeExplainer:
    model = build_model("resnet50", num_classes=4, stage=1)
    return LimeExplainer(model, device=torch.device("cpu"), num_samples=50, segments_n=30)


def test_lime_mask_shape_and_dtype() -> None:
    """LIME returns a (224, 224) boolean mask plus a positive time."""
    explainer = _explainer()
    rng = np.random.default_rng(0)
    image = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    mask, elapsed_ms = explainer.explain(image, target_class=0)
    assert mask.shape == (224, 224)
    assert mask.dtype == bool
    assert elapsed_ms > 0.0


def test_lime_accepts_grayscale_2d_input() -> None:
    """A 2-D grayscale array is stacked to 3 channels instead of crashing."""
    explainer = _explainer()
    rng = np.random.default_rng(1)
    gray = (rng.random((224, 224)) * 255).astype(np.uint8)
    mask, _ = explainer.explain(gray, target_class=2)
    assert mask.shape == (224, 224)
