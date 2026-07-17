"""Saliency overlay helpers for the demo (Phase 4).

The three XAI techniques produce different raw outputs (Grad-CAM a float map,
LIME a boolean mask, SHAP a float map), but all reduce to a single [0, 1]
saliency map that ``overlay_saliency`` blends over the original MRI. Keeping
these pure and dependency-light makes them trivially testable and reusable by
both the batch pipeline and the Gradio app.
"""

from __future__ import annotations

import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


def overlay_saliency(rgb_float: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    """Blend a [0, 1] saliency map over an image.

    Args:
        rgb_float: ``(H, W, 3)`` float array in [0, 1] — the original image.
        saliency: ``(H, W)`` float array in [0, 1] — the map to overlay.

    Returns:
        ``(H, W, 3)`` uint8 image with the heatmap blended in.
    """
    return show_cam_on_image(rgb_float, saliency.astype(np.float32), use_rgb=True)


def lime_mask_to_map(mask: np.ndarray) -> np.ndarray:
    """Convert a LIME boolean superpixel mask to a [0, 1] float map for overlay."""
    return mask.astype(np.float32)
