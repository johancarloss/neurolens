"""Tests for src/neurolens/ui/gradio_app.py — UI wiring, no models loaded.

``build_demo`` only reads ``inference.archs`` at construction time (the heavy
``explain`` runs on click), so we can validate the layout with a fake inference
that never loads a model.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from neurolens.ui.gradio_app import _format_times, _load_examples, build_demo

_EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "public" / "demo-examples"


class _FakeInference:
    """Stand-in exposing only what build_demo touches at construction time."""

    archs = ["vgg16", "resnet50"]


def test_load_examples_parses_all_six() -> None:
    """Every curated example yields a [path, story, stem] row with an existing image."""
    examples, _ = _load_examples(_EXAMPLES_DIR)
    assert len(examples) == 6
    for path, story, stem in examples:
        assert Path(path).exists()
        assert story.startswith("###")  # title heading
        assert stem  # non-empty cache key


def test_build_demo_constructs_blocks() -> None:
    """build_demo assembles a Gradio Blocks without loading any model."""
    demo = build_demo(_FakeInference(), examples_dir=_EXAMPLES_DIR)
    assert isinstance(demo, gr.Blocks)


def test_format_times_picks_readable_units() -> None:
    """Sub-second stays in ms; over a second switches to s; total is summed."""
    line = _format_times({"gradcam": 210.0, "lime": 15200.0, "shap": 3100.0})
    assert "Grad-CAM" in line
    assert "210 ms" in line  # sub-second -> ms
    assert "15.2 s" in line  # over a second -> seconds
    assert "18.5 s" in line  # total = 210 + 15200 + 3100 ms
