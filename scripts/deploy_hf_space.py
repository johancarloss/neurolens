"""Build and (optionally) deploy the NeuroLens HuggingFace Space (Phase 4, Bloco 6).

Assembles a self-contained Space directory — vendored ``neurolens`` package +
the two checkpoints + the curated examples (with pre-computed overlays) + a small
SHAP background (a mini ImageFolder) — and uploads it to a Gradio Space.

The HF token is read from the environment (``HF_TOKEN``) and never passed on the
command line or printed.

    # 1) assemble only (local smoke test, no token needed):
    uv run python scripts/deploy_hf_space.py --build-dir /tmp/neurolens-space

    # 2) assemble + push (needs HF_TOKEN in the environment):
    set -a; source .env; set +a
    uv run python scripts/deploy_hf_space.py --build-dir /tmp/neurolens-space --push
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CKPT_DIR = Path.home() / "neurolens-checkpoints-dataset"
DEFAULT_DATASET = Path.home() / "datasets" / "brain-tumor-mri"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# Curated example files must not double as SHAP background — keep the reference
# set distinct from the images we explain.
EXAMPLE_FILES = {
    "Te-gl_277.jpg",
    "Te-gl_113.jpg",
    "Te-me_101.jpg",
    "Te-me_169.jpg",
    "Te-pi_10.jpg",
    "Te-no_1.jpg",
}


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def assemble(
    build_dir: Path, ckpt_dir: Path, dataset_root: Path, background_per_class: int
) -> None:
    """Populate ``build_dir`` with everything the Space needs, self-contained."""
    build_dir.mkdir(parents=True, exist_ok=True)

    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(REPO_ROOT / "hf-space" / name, build_dir / name)

    _copy_tree(REPO_ROOT / "src" / "neurolens", build_dir / "neurolens")

    (build_dir / "checkpoints").mkdir(exist_ok=True)
    for arch in ("vgg16", "resnet50"):
        shutil.copy(
            ckpt_dir / f"{arch}_fold0_final.pt",
            build_dir / "checkpoints" / f"{arch}_fold0_final.pt",
        )

    _copy_tree(REPO_ROOT / "docs" / "public" / "demo-examples", build_dir / "demo-examples")

    background = build_dir / "background" / "Testing"
    for cls in CLASSES:
        out = background / cls
        out.mkdir(parents=True, exist_ok=True)
        picked = 0
        for img in sorted((dataset_root / "Testing" / cls).iterdir()):
            if img.name in EXAMPLE_FILES:
                continue
            shutil.copy(img, out / img.name)
            picked += 1
            if picked >= background_per_class:
                break

    print(f"assembled Space at {build_dir}")


def push(build_dir: Path) -> None:
    """Create the Space (if needed) and upload the assembled folder."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not in environment. Run `set -a; source .env; set +a` first.")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/neurolens"
    print(f"pushing to space {repo_id} ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(build_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy NeuroLens demo",
    )
    print(f"done: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/deploy the NeuroLens HF Space.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--background-per-class", type=int, default=5)
    parser.add_argument("--push", action="store_true", help="Upload after assembling.")
    args = parser.parse_args()

    assemble(args.build_dir, args.checkpoints_dir, args.dataset_root, args.background_per_class)
    if args.push:
        push(args.build_dir)


if __name__ == "__main__":
    main()
