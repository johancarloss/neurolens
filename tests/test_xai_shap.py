"""Tests for src/neurolens/xai/shap_explainer.py.

Integration-level on CPU. Uses a tiny background and ``nsamples`` to stay fast —
production uses 50 background images and nsamples=200.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from neurolens.models.factory import build_model
from neurolens.xai.shap_explainer import ShapExplainer


def _background_loader(n: int = 4) -> DataLoader:
    images = torch.randn(n, 3, 224, 224)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def test_shap_map_shape_and_range() -> None:
    """SHAP returns a (224, 224) map normalized to [0, 1] plus a positive time."""
    model = build_model("resnet50", num_classes=4, stage=1)
    explainer = ShapExplainer(
        model,
        background_loader=_background_loader(),
        device=torch.device("cpu"),
        num_background=4,
        nsamples=10,
    )
    shap_map, elapsed_ms = explainer.explain(torch.randn(1, 3, 224, 224))
    assert shap_map.shape == (224, 224)
    assert shap_map.min() >= 0.0 and shap_map.max() <= 1.0
    assert elapsed_ms > 0.0
