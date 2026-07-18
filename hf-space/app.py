"""HuggingFace Spaces entrypoint for the NeuroLens demo (Phase 4, Bloco 6).

Runs the same Gradio app as ``scripts/run_demo.py``, but with every path relative
to the Space repo: the two checkpoints, the curated examples (with their
pre-computed overlays), and a small SHAP background are all bundled alongside.
HuggingFace runs this file to launch the Space.
"""

from pathlib import Path

from neurolens.ui.gradio_app import build_demo
from neurolens.ui.inference import NeuroLensInference

ROOT = Path(__file__).parent

_checkpoints = {
    "vgg16": str(ROOT / "checkpoints" / "vgg16_fold0_final.pt"),
    "resnet50": str(ROOT / "checkpoints" / "resnet50_fold0_final.pt"),
}

# Loaded once at boot (models + 6 explainers + SHAP background) — see the
# cold-start note in the Phase 4 plan.
inference = NeuroLensInference(
    checkpoints=_checkpoints,
    dataset_root=ROOT / "background",
    lime_num_samples=200,
    shap_nsamples=100,
)
demo = build_demo(inference, examples_dir=ROOT / "demo-examples")

if __name__ == "__main__":
    demo.launch()
