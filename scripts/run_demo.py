"""Launch the NeuroLens Gradio demo locally (Phase 4).

Loads both architectures + their explainers and serves the interactive UI.
Defaults point at the local checkpoint dataset and MRI dataset; override for
other environments.

    uv run python scripts/run_demo.py            # local, http://127.0.0.1:7860
    uv run python scripts/run_demo.py --share    # public tunnel (72h)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neurolens.ui.gradio_app import build_demo
from neurolens.ui.inference import NeuroLensInference

DEFAULT_CKPT_DIR = Path.home() / "neurolens-checkpoints-dataset"
DEFAULT_DATASET = Path.home() / "datasets" / "brain-tumor-mri"
DEFAULT_EXAMPLES = Path(__file__).parent.parent / "docs" / "public" / "demo-examples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeuroLens Gradio demo.")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Public Gradio tunnel (72h)")
    parser.add_argument("--lime-samples", type=int, default=200)
    parser.add_argument("--shap-samples", type=int, default=100)
    args = parser.parse_args()

    checkpoints = {
        "vgg16": str(args.checkpoints_dir / "vgg16_fold0_final.pt"),
        "resnet50": str(args.checkpoints_dir / "resnet50_fold0_final.pt"),
    }

    inference = NeuroLensInference(
        checkpoints=checkpoints,
        dataset_root=args.dataset_root,
        lime_num_samples=args.lime_samples,
        shap_nsamples=args.shap_samples,
    )
    demo = build_demo(inference, examples_dir=args.examples_dir)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
