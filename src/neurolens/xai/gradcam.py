"""Grad-CAM explainer (Phase 3).

Grad-CAM answers: *"which regions of the last convolutional block most drove
the predicted class?"* It weights the final feature maps by the gradient of the
class score flowing into them, then upsamples to a 224x224 heatmap.

This wrapper delegates the heavy lifting to ``pytorch-grad-cam`` and uses the
arch-aware target layer registered in the factory (the bridge built in Phase 2:
VGG16 -> ``features[-1]``, ResNet50 -> ``layer4[-1]``).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from neurolens.models.factory import get_target_layer_for_gradcam


class GradCAMExplainer:
    """Generate Grad-CAM saliency maps for a trained classifier."""

    def __init__(self, model: nn.Module, arch: str, device: torch.device) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.target_layer = get_target_layer_for_gradcam(model, arch)

    def explain(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, float]:
        """Generate a Grad-CAM map for a single image.

        Args:
            input_tensor: ``(1, 3, 224, 224)`` normalized tensor on ``self.device``.
            target_class: class index to explain; ``None`` uses the model's argmax.

        Returns:
            ``(grayscale_cam, compute_time_ms)`` where ``grayscale_cam`` is a
            ``(224, 224)`` float array in [0, 1].
        """
        if target_class is None:
            with torch.no_grad():
                target_class = int(self.model(input_tensor).argmax(1).item())

        targets = [ClassifierOutputTarget(target_class)]
        start = time.perf_counter()
        with GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return grayscale_cam, elapsed_ms

    def overlay(self, rgb_image: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
        """Overlay a CAM heatmap on the original image.

        Args:
            rgb_image: ``(H, W, 3)`` float array in [0, 1].
            grayscale_cam: ``(H, W)`` float array in [0, 1].

        Returns:
            ``(H, W, 3)`` uint8 image with the heatmap blended in.
        """
        return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
