"""Pre-compute the curated demo examples offline (Phase 4, Bloco 4).

Runs every example in ``examples.yaml`` through both architectures + the three
explainers once, and persists the overlays so the demo can serve them instantly.
Re-run whenever the examples, models, or explainer settings change.

    uv run python scripts/precompute_examples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from neurolens.ui.inference import NeuroLensInference
from neurolens.ui.precompute import result_dir, save_result

DEFAULT_CKPT_DIR = Path.home() / "neurolens-checkpoints-dataset"
DEFAULT_DATASET = Path.home() / "datasets" / "brain-tumor-mri"
DEFAULT_EXAMPLES = Path(__file__).parent.parent / "docs" / "public" / "demo-examples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute demo example overlays.")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES)
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

    spec = yaml.safe_load((args.examples_dir / "examples.yaml").read_text())
    precomputed_root = args.examples_dir / "precomputed"
    examples = spec["examples"]

    for i, ex in enumerate(examples, start=1):
        stem = Path(ex["file"]).stem
        image = np.array(Image.open(args.examples_dir / ex["file"]).convert("RGB"))
        print(f"[{i}/{len(examples)}] computing {stem} ...", flush=True)
        result = inference.explain(image)
        save_result(result, result_dir(precomputed_root, stem))
        print(f"    saved -> {result_dir(precomputed_root, stem)}", flush=True)

    print(f"done: {len(examples)} examples pre-computed into {precomputed_root}")


if __name__ == "__main__":
    main()
