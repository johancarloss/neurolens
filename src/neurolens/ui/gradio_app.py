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
"""

_UPLOAD_STORY = "*Your own scan — click **Classify & Explain**. Live inference takes ~20–60s.*"


def _load_examples(examples_dir: Path) -> tuple[list[list[str]], dict[str, str]]:
    """Return (gradio_examples, story_by_path) from examples.yaml.

    gradio_examples is ``[[image_path, story_markdown], ...]`` so clicking an
    example fills both the image input and the narrative panel.
    """
    spec = yaml.safe_load((examples_dir / "examples.yaml").read_text())
    gradio_examples: list[list[str]] = []
    for ex in spec["examples"]:
        path = str(examples_dir / ex["file"])
        story = f"### {ex['title']}\n\n{ex['story']}"
        gradio_examples.append([path, story])
    return gradio_examples, {}


def build_demo(inference: NeuroLensInference, examples_dir: str | Path) -> gr.Blocks:
    """Assemble the Gradio Blocks demo over a ready ``NeuroLensInference``."""
    examples_dir = Path(examples_dir)
    gradio_examples, _ = _load_examples(examples_dir)
    archs = inference.archs

    def run(image: np.ndarray | None) -> list[object]:
        """Classify + explain, returning the flat list of outputs Gradio expects."""
        if image is None:
            return [None] + [None] * (4 * len(archs))
        result = inference.explain(image)
        outputs: list[object] = [result.original]
        for arch in archs:
            r = result.per_arch[arch]
            outputs.extend([r.probs, r.gradcam, r.lime, r.shap])
        return outputs

    with gr.Blocks(title="NeuroLens — Brain Tumor MRI Classifier", fill_width=True) as demo:
        gr.Markdown(_HEADER)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(type="numpy", label="MRI scan", height=320)
                run_btn = gr.Button("Classify & Explain", variant="primary")
            with gr.Column(scale=3):
                original_view = gr.Image(label="Original (reference)", height=320)
                story_md = gr.Markdown(_UPLOAD_STORY)

        gr.Examples(
            examples=gradio_examples,
            inputs=[image_input, story_md],
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
                    arch_outputs.extend([label, gc, li, sh])

        gr.Markdown(
            "*Grad-CAM = where the model's gradients point · LIME = which regions, "
            "if hidden, flip the decision · SHAP = each pixel's contribution. "
            "They disagree by design — that's why we show all three.*"
        )

        # Free uploads have no preset story; reset the narrative panel.
        image_input.upload(lambda: _UPLOAD_STORY, outputs=story_md)

        run_btn.click(run, inputs=image_input, outputs=[original_view, *arch_outputs])

    return demo
