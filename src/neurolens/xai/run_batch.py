"""XAI batch orchestrator (Phase 3) — the ``xai_batch`` job entry point.

Loads each trained architecture, picks the target images (glioma-focused
stratified selection), runs Grad-CAM + LIME + SHAP on each, computes the
comparison metrics, saves overlays, and persists everything to PostgreSQL.

Both architectures are explained on the SAME images so the analysis can compare
where each model looks. Artifacts/comparisons link to the prediction of the
exact (arch, fold) being explained (``XaiArchSpec.run_id``).

Invoked by the universal Kaggle runner via ``main(config_profile=...)``, which
resolves ``configs/{profile}.yaml`` (e.g. ``xai_default``, ``xai_smoke``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torch.utils.data import DataLoader

from neurolens.config import XaiArchSpec, XaiConfig, load_xai_config
from neurolens.data.dataset import build_dataset
from neurolens.data.kaggle_paths import discover_brain_tumor_dataset
from neurolens.data.transforms import eval_transforms
from neurolens.db.repository import (
    get_prediction_id,
    insert_xai_artifact,
    insert_xai_comparison,
)
from neurolens.models.factory import build_model
from neurolens.xai.gradcam import GradCAMExplainer
from neurolens.xai.lime_explainer import LimeExplainer
from neurolens.xai.metrics import compute_all_metrics
from neurolens.xai.selection import select_images
from neurolens.xai.shap_explainer import ShapExplainer


def _resolve_checkpoint(checkpoint: str) -> str:
    """Return the checkpoint path, falling back to a search under /kaggle/input.

    Kaggle mounts datasets at unpredictable paths (the brain-tumor set lands at
    ``/kaggle/input/datasets/<owner>/<slug>``, not the documented
    ``/kaggle/input/<slug>``), so a hardcoded path is fragile. If the configured
    path is absent, search by filename — the same robustness kaggle_paths uses.
    """
    if Path(checkpoint).exists():
        return checkpoint
    name = Path(checkpoint).name
    kaggle_input = Path("/kaggle/input")
    matches = sorted(kaggle_input.rglob(name)) if kaggle_input.exists() else []
    if matches:
        print(f"[run_batch] resolved checkpoint '{name}' -> {matches[0]}")
        return str(matches[0])
    raise FileNotFoundError(
        f"Checkpoint '{name}' not found at '{checkpoint}' nor under /kaggle/input. "
        f"Is the neurolens-checkpoints dataset attached to the kernel?"
    )


def _load_model(arch: str, checkpoint: str, device: torch.device) -> nn.Module:
    """Build the architecture (stage 2 layout) and load its fine-tuned weights."""
    model = build_model(arch, num_classes=4, stage=2)
    state = torch.load(_resolve_checkpoint(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def _prepare_image(image_path: str, image_size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    """Return ``(input_tensor (1,3,H,W), rgb_uint8 (H,W,3))`` for one image."""
    pil = Image.open(image_path).convert("RGB")
    rgb_uint8 = np.array(pil.resize((image_size, image_size)))
    input_tensor = eval_transforms(image_size=image_size)(pil).unsqueeze(0)
    return input_tensor, rgb_uint8


def _save_overlay(rgb_float: np.ndarray, saliency: np.ndarray, path: Path) -> None:
    """Save a [0,1] saliency map blended over the image (shared by all 3 methods)."""
    overlay = show_cam_on_image(rgb_float, saliency.astype(np.float32), use_rgb=True)
    Image.fromarray(overlay).save(path)


def _explain_one_image(
    sel: Any,
    arch_spec: XaiArchSpec,
    explainers: dict[str, Any],
    filename_to_path: dict[str, str],
    class_to_idx: dict[str, int],
    config: XaiConfig,
    output_root: Path,
    device: torch.device,
) -> None:
    """Run the 3 techniques on one image for one arch and persist the results."""
    prediction_id = get_prediction_id(arch_spec.run_id, sel.image_filename)
    image_path = filename_to_path.get(sel.image_filename)
    if prediction_id is None or image_path is None:
        print(f"[run_batch] skip {sel.image_filename} (no prediction_id/path for {arch_spec.name})")
        return

    input_tensor, rgb_uint8 = _prepare_image(image_path)
    input_tensor = input_tensor.to(device)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    target_class = class_to_idx[sel.predicted_label]  # explain the predicted class
    prefix = f"{arch_spec.name}_{prediction_id}"

    # Grad-CAM
    gc_map, gc_ms = explainers["gradcam"].explain(input_tensor, target_class)
    _save_overlay(rgb_float, gc_map, output_root / f"{prefix}_gradcam.png")

    # LIME (repeated for the stability metric; first run is the saved overlay)
    lime_masks: list[np.ndarray] = []
    lime_ms = 0.0
    for _ in range(config.lime_stability_runs):
        mask, lime_ms = explainers["lime"].explain(rgb_uint8, target_class)
        lime_masks.append(mask)
    _save_overlay(rgb_float, lime_masks[0].astype(np.float32), output_root / f"{prefix}_lime.png")

    # SHAP
    shap_map, shap_ms = explainers["shap"].explain(input_tensor, target_class)
    _save_overlay(rgb_float, shap_map, output_root / f"{prefix}_shap.png")

    metrics = compute_all_metrics(
        gc_map,
        lime_masks[0],
        shap_map,
        lime_masks_for_stability=lime_masks,
        times={"time_ms_gradcam": gc_ms, "time_ms_lime": lime_ms, "time_ms_shap": shap_ms},
        threshold=config.binarization_threshold,
    )
    metrics["binarization_threshold"] = config.binarization_threshold

    try:
        for method in ("gradcam", "lime", "shap"):
            insert_xai_artifact(
                prediction_id=prediction_id,
                method=method,
                target_class=sel.predicted_label,
                artifact_path=f"{prefix}_{method}.png",
                compute_time_ms=metrics[f"time_ms_{method}"],
                metadata={"arch": arch_spec.name, "reason": sel.reason},
            )
        insert_xai_comparison(prediction_id, metrics)
    except Exception as exc:  # noqa: BLE001 — never break the batch on a logging fail
        print(f"[run_batch] WARNING: persist failed for {prefix}: {exc!r}")


def main(config_profile: str | None = None) -> None:
    """Entry point invoked by the Kaggle runner. Resolves ``configs/{profile}.yaml``."""
    repo_dir = Path("/kaggle/working/neurolens-repo")
    profile = config_profile or os.environ.get("CONFIG_PROFILE") or "xai_default"
    config = load_xai_config(repo_dir / "configs" / f"{profile}.yaml")
    print(f"[run_batch] profile={profile} archs={[a.name for a in config.archs]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_batch] device={device}")

    paths = discover_brain_tumor_dataset()
    test_dataset = build_dataset(paths.data_root, transform=eval_transforms(), split="test")
    filename_to_path = {Path(p).name: p for p, _ in test_dataset.samples}
    class_to_idx = test_dataset.class_to_idx

    selected = select_images(config.selection_run_ids, config.num_images)
    print(f"[run_batch] selected {len(selected)} images")

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for arch_spec in config.archs:
        print(f"[run_batch] === {arch_spec.name} (run {arch_spec.run_id}) ===")
        model = _load_model(arch_spec.name, arch_spec.checkpoint, device)
        explainers = {
            "gradcam": GradCAMExplainer(model, arch_spec.name, device),
            "lime": LimeExplainer(
                model,
                device,
                num_samples=config.lime_num_samples,
                segments_n=config.lime_segments_n,
            ),
            "shap": ShapExplainer(
                model,
                DataLoader(test_dataset, batch_size=32),
                device,
                num_background=config.shap_num_background,
                nsamples=config.shap_nsamples,
            ),
        }
        for sel in selected:
            _explain_one_image(
                sel,
                arch_spec,
                explainers,
                filename_to_path,
                class_to_idx,
                config,
                output_root,
                device,
            )

    print("[run_batch] done")
