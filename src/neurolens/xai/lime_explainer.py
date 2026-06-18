"""LIME explainer (Phase 3).

LIME answers: *"if I hide this region, does the model change its mind?"* It
splits the image into superpixels (SLIC), generates many perturbed copies with
random superpixels switched off, asks the model to classify each, and fits a
simple linear model to learn which superpixels matter for the target class.

Two project-specific notes:
- MRI is grayscale; LIME/SLIC expect RGB, so a 1-channel image is stacked to 3.
- LIME is stochastic and slow (it runs the model on ``num_samples`` perturbed
  images), which is why Phase 3 caps the image count and measures stability.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch import nn

from neurolens.data.transforms import IMAGENET_MEAN, IMAGENET_STD


class LimeExplainer:
    """Generate LIME superpixel masks for a trained classifier."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_samples: int = 1000,
        segments_n: int = 100,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.num_samples = num_samples
        self.explainer = lime_image.LimeImageExplainer()
        self.segmenter = SegmentationAlgorithm(
            "slic", n_segments=segments_n, compactness=10, sigma=1
        )

    def _batch_predict(self, images: np.ndarray) -> np.ndarray:
        """Classifier callback LIME calls with ``(N, H, W, 3)`` uint8 images.

        Normalizes the batch the same way training did and returns ``(N, 4)``
        class probabilities.
        """
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
        images_norm = (images.astype(np.float32) / 255.0 - mean) / std
        images_norm = images_norm.transpose(0, 3, 1, 2)  # (N, 3, H, W)
        x = torch.from_numpy(images_norm).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
        return probs.cpu().numpy()

    def explain(self, rgb_image_uint8: np.ndarray, target_class: int) -> tuple[np.ndarray, float]:
        """Generate a LIME mask for one image.

        Args:
            rgb_image_uint8: ``(H, W, 3)`` uint8 (a 2-D grayscale array is
                stacked to 3 channels automatically).
            target_class: class index to explain.

        Returns:
            ``(mask, compute_time_ms)`` where ``mask`` is a ``(H, W)`` boolean
            array marking the superpixels supporting ``target_class``.
        """
        if rgb_image_uint8.ndim == 2:
            rgb_image_uint8 = np.stack([rgb_image_uint8] * 3, axis=-1)

        start = time.perf_counter()
        explanation = self.explainer.explain_instance(
            rgb_image_uint8,
            classifier_fn=self._batch_predict,
            top_labels=4,
            hide_color=0,
            num_samples=self.num_samples,
            segmentation_fn=self.segmenter,
            batch_size=32,
        )
        _, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return mask.astype(bool), elapsed_ms
