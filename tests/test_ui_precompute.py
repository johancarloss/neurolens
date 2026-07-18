"""Tests for src/neurolens/ui/precompute.py — the pre-computed example cache.

No models are loaded; we build a synthetic ExplainResult, round-trip it through
disk, and assert the arrays and numbers survive exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurolens.ui.inference import ArchResult, ExplainResult
from neurolens.ui.precompute import load_result, result_dir, save_result

_ARCHS = ["vgg16", "resnet50"]


def _fake_arch(arch: str) -> ArchResult:
    return ArchResult(
        arch=arch,
        probs={"glioma": 0.7, "meningioma": 0.1, "notumor": 0.1, "pituitary": 0.1},
        predicted_label="glioma",
        gradcam=np.full((4, 4, 3), 10, dtype=np.uint8),
        lime=np.full((4, 4, 3), 20, dtype=np.uint8),
        shap=np.full((4, 4, 3), 30, dtype=np.uint8),
        times_ms={"gradcam": 1.0, "lime": 2.0, "shap": 3.0},
    )


def _fake_result() -> ExplainResult:
    original = np.full((4, 4, 3), 100, dtype=np.uint8)
    return ExplainResult(original=original, per_arch={a: _fake_arch(a) for a in _ARCHS})


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Arrays and numbers survive a save -> load cycle exactly."""
    out = result_dir(tmp_path, "Te-gl_277")
    save_result(_fake_result(), out)

    loaded = load_result(out, _ARCHS)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.original, np.full((4, 4, 3), 100, dtype=np.uint8))
    for arch in _ARCHS:
        r = loaded.per_arch[arch]
        assert r.predicted_label == "glioma"
        assert r.probs["glioma"] == 0.7
        assert r.times_ms["lime"] == 2.0
        np.testing.assert_array_equal(r.gradcam, np.full((4, 4, 3), 10, dtype=np.uint8))


def test_missing_cache_returns_none(tmp_path: Path) -> None:
    """A missing directory yields None so the UI can fall back to a live run."""
    assert load_result(tmp_path / "does-not-exist", _ARCHS) is None


def test_incomplete_cache_returns_none(tmp_path: Path) -> None:
    """A cache missing one architecture's maps is treated as absent."""
    out = result_dir(tmp_path, "partial")
    save_result(_fake_result(), out)
    (out / "resnet50_lime.png").unlink()  # corrupt the cache
    assert load_result(out, _ARCHS) is None
