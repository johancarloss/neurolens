"""Inference layer for the demo (Phase 4) — logic separated from the Gradio UI.

``NeuroLensInference`` loads both trained architectures and their three
explainers ONCE (the models and SHAP background are expensive), then exposes a
single ``explain(image)`` call that returns predictions + overlays for both
architectures. Keeping this independent of Gradio makes it unit-testable and
reusable (CLI, notebooks, the app).

Design notes:
- Models are loaded at ``stage=2`` (the fine-tuned layout the ``.pt`` was saved
  from) and every parameter is left trainable so Grad-CAM/SHAP gradients flow.
- SHAP needs a fixed background set (expected-gradients baseline), so the demo
  is NOT stateless per image — a handful of dataset images are loaded at boot.
- Demo speed profile (vs research): LIME 200 samples, SHAP 100, no stability
  runs — good enough visually, ~10x faster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from neurolens.data.dataset import CLASSES, build_dataset
from neurolens.data.transforms import eval_transforms
from neurolens.models.factory import build_model
from neurolens.ui.visualize import lime_mask_to_map, overlay_saliency
from neurolens.xai.gradcam import GradCAMExplainer
from neurolens.xai.lime_explainer import LimeExplainer
from neurolens.xai.shap_explainer import ShapExplainer


@dataclass
class ArchResult:
    """One architecture's explanation of one image."""

    arch: str
    probs: dict[str, float]
    predicted_label: str
    gradcam: np.ndarray  # (H, W, 3) uint8 overlay
    lime: np.ndarray
    shap: np.ndarray
    times_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ExplainResult:
    """Full result for one image: the original + one ArchResult per architecture."""

    original: np.ndarray  # (H, W, 3) uint8, resized to model input
    per_arch: dict[str, ArchResult]


class NeuroLensInference:
    """Holds both models + their explainers and runs the full XAI comparison."""

    def __init__(
        self,
        checkpoints: dict[str, str],
        dataset_root: str | Path,
        device: torch.device | None = None,
        image_size: int = 224,
        lime_num_samples: int = 200,
        shap_nsamples: int = 100,
        shap_num_background: int = 16,
    ) -> None:
        """Load both architectures and their three explainers once.

        Args:
            checkpoints: ``{"vgg16": path, "resnet50": path}`` to the ``.pt`` files.
            dataset_root: dataset root (has ``Testing/``) — used for the SHAP
                background set only.
            device: torch device; defaults to CUDA if available, else CPU.
            image_size: model input size (224).
            lime_num_samples / shap_nsamples / shap_num_background: demo-speed
                budgets (smaller than research).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self._transform = eval_transforms(image_size=image_size)

        background_loader = self._build_background_loader(dataset_root)

        self._explainers: dict[str, dict[str, object]] = {}
        for arch, ckpt in checkpoints.items():
            model = self._load_model(arch, ckpt)
            self._explainers[arch] = {
                "model": model,
                "gradcam": GradCAMExplainer(model, arch=arch, device=self.device),
                "lime": LimeExplainer(model, self.device, num_samples=lime_num_samples),
                "shap": ShapExplainer(
                    model,
                    background_loader,
                    self.device,
                    num_background=shap_num_background,
                    nsamples=shap_nsamples,
                ),
            }

    @property
    def archs(self) -> list[str]:
        """Architectures loaded, in insertion order."""
        return list(self._explainers)

    def _load_model(self, arch: str, checkpoint: str) -> nn.Module:
        model = build_model(arch, num_classes=len(CLASSES), stage=2)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        return model.eval().to(self.device)

    def _build_background_loader(self, dataset_root: str | Path) -> DataLoader:
        test_dataset = build_dataset(dataset_root, transform=self._transform, split="test")
        return DataLoader(test_dataset, batch_size=8, shuffle=True)

    def _prepare(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Return ``(input_tensor (1,3,H,W), rgb_uint8 (H,W,3))`` for one image."""
        pil = Image.fromarray(image_rgb).convert("RGB").resize((self.image_size, self.image_size))
        rgb_uint8 = np.array(pil)
        input_tensor = self._transform(pil).unsqueeze(0).to(self.device)
        return input_tensor, rgb_uint8

    def explain(self, image_rgb: np.ndarray) -> ExplainResult:
        """Run prediction + the 3 techniques for every architecture on one image."""
        input_tensor, rgb_uint8 = self._prepare(image_rgb)
        rgb_float = rgb_uint8.astype(np.float32) / 255.0

        per_arch: dict[str, ArchResult] = {}
        for arch, ex in self._explainers.items():
            model = ex["model"]
            with torch.no_grad():
                probs_t = torch.softmax(model(input_tensor), dim=1)[0]
            probs = {cls: float(probs_t[i]) for i, cls in enumerate(CLASSES)}
            target_class = int(probs_t.argmax())
            predicted_label = CLASSES[target_class]

            gc_map, gc_ms = ex["gradcam"].explain(input_tensor, target_class)
            lime_mask, lime_ms = ex["lime"].explain(rgb_uint8, target_class)
            shap_map, shap_ms = ex["shap"].explain(input_tensor, target_class)

            per_arch[arch] = ArchResult(
                arch=arch,
                probs=probs,
                predicted_label=predicted_label,
                gradcam=overlay_saliency(rgb_float, gc_map),
                lime=overlay_saliency(rgb_float, lime_mask_to_map(lime_mask)),
                shap=overlay_saliency(rgb_float, shap_map),
                times_ms={"gradcam": gc_ms, "lime": lime_ms, "shap": shap_ms},
            )

        return ExplainResult(original=rgb_uint8, per_arch=per_arch)
