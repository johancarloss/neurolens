"""Gradio interface for the NeuroLens demo (Phase 4).

The UI is a thin shell over ``NeuroLensInference`` (which holds the models and
explainers). It shows the two architectures **side by side** (so the "they agree
on the class but look in different places" finding is visible at a glance), with
the original MRI as the shared reference and the curated examples carrying the
scientific narrative.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from neurolens.ui.inference import NeuroLensInference
from neurolens.ui.precompute import load_result

_ARCH_LABEL = {"vgg16": "VGG16", "resnet50": "ResNet50"}

_HEADER = """
# 🧠 NeuroLens — Brain Tumor MRI Classifier with Explainability

Upload a brain MRI (or pick a curated example) and see **two architectures**
(VGG16 & ResNet50) classify it — each explained by **three XAI techniques**
(Grad-CAM, LIME, SHAP), side by side against the original scan.

This is the interactive companion to a study that found the two models *agree on
their predictions but look in different places*, and that both fail on gliomas
for the same, data-driven reason. Try `Te-gl_277` to see a glioma both models
miss — and where they (wrongly) look.

📄 [Code & methodology](https://github.com/johancarloss/neurolens) ·
[Phase 3 — the XAI analysis](https://github.com/johancarloss/neurolens/blob/main/docs/public/phases/phase-3-xai.md) ·
[How Grad-CAM, LIME and SHAP work](https://github.com/johancarloss/neurolens/blob/main/docs/public/methodology/explainability.md)
"""

_UPLOAD_STORY = (
    "*Your own scan — click **Classify & Explain**. Live inference takes "
    "**~4 minutes** on CPU (LIME and SHAP dominate); the curated examples above "
    "are pre-computed and open instantly.*"
)

_TIMES_PLACEHOLDER = "*Compute times appear here after you classify.*"

_TIMES_ORDER = [("gradcam", "Grad-CAM"), ("lime", "LIME"), ("shap", "SHAP")]


def _human_ms(ms: float) -> str:
    """Render a duration in the most readable unit (ms under a second, else s)."""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _format_times(times_ms: dict[str, float], precomputed: bool = False) -> str:
    """Render per-technique compute times as one compact markdown line.

    Makes the cost of explainability visible: Grad-CAM is ~1 backward pass while
    LIME/SHAP run the model hundreds of times, so their times differ by orders of
    magnitude — a point worth showing in a defense. Curated examples load
    instantly from cache, so their line is labelled *compute cost (pre-computed)*
    to make clear the numbers are the offline compute cost, not the user's wait.
    """
    parts = [
        f"**{label}** {_human_ms(times_ms[key])}" for key, label in _TIMES_ORDER if key in times_ms
    ]
    total = sum(times_ms.values())
    parts.append(f"**total** {_human_ms(total)}")
    prefix = "⏱️ compute cost (pre-computed): " if precomputed else "⏱️ "
    return prefix + " · ".join(parts)


def _load_examples(examples_dir: Path) -> tuple[list[list[str]], list[str]]:
    """Return (gradio_examples, example_labels) from examples.yaml.

    gradio_examples is ``[[image_path, story_markdown, stem], ...]``: clicking an
    example fills the image input, the narrative panel, and a hidden field with
    the example's stem (used to load its pre-computed maps). example_labels are
    the titles shown under each thumbnail so users can tell them apart.
    """
    spec = yaml.safe_load((examples_dir / "examples.yaml").read_text())
    gradio_examples: list[list[str]] = []
    labels: list[str] = []
    for ex in spec["examples"]:
        path = str(examples_dir / ex["file"])
        story = f"### {ex['title']}\n\n{ex['story']}"
        stem = Path(ex["file"]).stem
        gradio_examples.append([path, story, stem])
        labels.append(ex["title"])
    return gradio_examples, labels


def build_demo(inference: NeuroLensInference, examples_dir: str | Path) -> gr.Blocks:
    """Assemble the Gradio Blocks demo over a ready ``NeuroLensInference``."""
    examples_dir = Path(examples_dir)
    precomputed_root = examples_dir / "precomputed"
    gradio_examples, example_labels = _load_examples(examples_dir)
    archs = inference.archs

    def run(image: np.ndarray | None, example_id: str) -> list[object]:
        """Classify + explain, returning the flat list of outputs Gradio expects.

        Curated examples (``example_id`` set) load pre-computed maps instantly;
        free uploads compute live (~4 min on CPU, measured).
        """
        if image is None:
            return [None] + [None] * (5 * len(archs))
        result = load_result(precomputed_root / example_id, archs) if example_id else None
        precomputed = result is not None
        if result is None:
            result = inference.explain(image)
        outputs: list[object] = [result.original]
        for arch in archs:
            r = result.per_arch[arch]
            outputs.extend(
                [
                    r.probs,
                    r.gradcam,
                    r.lime,
                    r.shap,
                    _format_times(r.times_ms, precomputed=precomputed),
                ]
            )
        return outputs

    with gr.Blocks(title="NeuroLens — Brain Tumor MRI Classifier", fill_width=True) as demo:
        gr.Markdown(_HEADER)

        # Hidden: which curated example is loaded (empty for free uploads).
        # Routes run() to the pre-computed cache instead of a live compute.
        selected_example = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(type="numpy", label="MRI scan", height=320)
                run_btn = gr.Button("Classify & Explain", variant="primary")
            with gr.Column(scale=3):
                original_view = gr.Image(label="Original (reference)", height=320)
                story_md = gr.Markdown(_UPLOAD_STORY)

        gr.Examples(
            examples=gradio_examples,
            inputs=[image_input, story_md, selected_example],
            example_labels=example_labels,
            label="Curated examples — each illustrates a finding (click to load)",
        )

        gr.Markdown("## Predictions & explanations — the two architectures side by side")
        with gr.Row():
            arch_outputs: list[object] = []
            for arch in archs:
                with gr.Column():
                    gr.Markdown(f"### {_ARCH_LABEL.get(arch, arch)}")
                    label = gr.Label(num_top_classes=4, label="Prediction")
                    with gr.Row():
                        gc = gr.Image(label="Grad-CAM", height=200)
                        li = gr.Image(label="LIME", height=200)
                        sh = gr.Image(label="SHAP", height=200)
                    times = gr.Markdown(_TIMES_PLACEHOLDER)
                    arch_outputs.extend([label, gc, li, sh, times])

        gr.Markdown(
            "*Grad-CAM = where the model's gradients point · LIME = which regions, "
            "if hidden, flip the decision · SHAP = each pixel's contribution. "
            "They disagree by design — that's why we show all three.*"
        )

        # A free upload is not a curated example: clear the story + the cache key.
        image_input.upload(lambda: ("", _UPLOAD_STORY), outputs=[selected_example, story_md])

        run_btn.click(
            run, inputs=[image_input, selected_example], outputs=[original_view, *arch_outputs]
        )

    return demo
