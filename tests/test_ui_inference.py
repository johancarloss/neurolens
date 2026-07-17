"""Tests for src/neurolens/ui/visualize.py and inference.py.

The visualize helpers are pure and always run. The inference test is an
integration check that needs the local checkpoints + dataset, so it is skipped
where those are absent (e.g. CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolens.data.dataset import CLASSES
from neurolens.ui.visualize import lime_mask_to_map, overlay_saliency

_CKPT_DIR = Path.home() / "neurolens-checkpoints-dataset"
_DATASET = Path.home() / "datasets" / "brain-tumor-mri"
_RESNET_CKPT = _CKPT_DIR / "resnet50_fold0_final.pt"
_HAS_LOCAL_ASSETS = _RESNET_CKPT.exists() and _DATASET.exists()


def test_overlay_saliency_shape_and_dtype() -> None:
    """Blending a [0,1] map over an image returns (H, W, 3) uint8."""
    rng = np.random.default_rng(0)
    rgb = rng.random((224, 224, 3)).astype(np.float32)
    saliency = rng.random((224, 224)).astype(np.float32)
    out = overlay_saliency(rgb, saliency)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_lime_mask_to_map_converts_bool_to_float() -> None:
    """A boolean LIME mask becomes a float map with the same True/False layout."""
    mask = np.array([[True, False], [False, True]])
    result = lime_mask_to_map(mask)
    assert result.dtype == np.float32
    assert result.tolist() == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.skipif(not _HAS_LOCAL_ASSETS, reason="local checkpoints/dataset not present")
def test_inference_explain_structure() -> None:
    """explain() returns a full result for each architecture, with valid probs."""
    import torch

    from neurolens.ui.inference import NeuroLensInference

    inference = NeuroLensInference(
        checkpoints={"resnet50": str(_RESNET_CKPT)},
        dataset_root=_DATASET,
        device=torch.device("cpu"),
        lime_num_samples=20,
        shap_nsamples=5,
        shap_num_background=4,
    )
    image = (np.random.default_rng(1).random((224, 224, 3)) * 255).astype(np.uint8)
    result = inference.explain(image)

    assert result.original.shape == (224, 224, 3)
    assert set(result.per_arch) == {"resnet50"}

    arch_result = result.per_arch["resnet50"]
    assert arch_result.predicted_label in CLASSES
    assert set(arch_result.probs) == set(CLASSES)
    assert abs(sum(arch_result.probs.values()) - 1.0) < 1e-4
    for overlay in (arch_result.gradcam, arch_result.lime, arch_result.shap):
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8
