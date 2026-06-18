"""SHAP explainer (Phase 3).

SHAP answers: *"what is each pixel's fair contribution to the prediction?"* —
grounded in game theory (Shapley values). We use ``GradientExplainer``, which
estimates contributions from gradients along paths between the input and a set
of background images.

Why GradientExplainer (not DeepExplainer): DeepExplainer breaks on recent
PyTorch versions (decision D7 in the Phase 3 plan). GradientExplainer is the
robust choice for our Torch stack.
"""

from __future__ import annotations

import time

import numpy as np
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader


class ShapExplainer:
    """Generate SHAP saliency maps for a trained classifier."""

    def __init__(
        self,
        model: nn.Module,
        background_loader: DataLoader,
        device: torch.device,
        num_background: int = 50,
        nsamples: int = 200,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.nsamples = nsamples

        # Collect a fixed background set for the expected-gradients baseline.
        background_list: list[torch.Tensor] = []
        collected = 0
        for x, _ in background_loader:
            background_list.append(x)
            collected += x.size(0)
            if collected >= num_background:
                break
        background = torch.cat(background_list, dim=0)[:num_background].to(device)
        self.explainer = shap.GradientExplainer(self.model, background)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,  # noqa: ARG002 — kept for explainer API symmetry
    ) -> tuple[np.ndarray, float]:
        """Generate a SHAP saliency map for a single image.

        Args:
            input_tensor: ``(1, 3, 224, 224)`` normalized tensor.
            target_class: kept for API symmetry; GradientExplainer with
                ``ranked_outputs=1`` already targets the top-predicted class.

        Returns:
            ``(shap_map, compute_time_ms)`` where ``shap_map`` is a
            ``(224, 224)`` float array in [0, 1] (abs contributions, summed over
            the 3 channels and normalized).
        """
        start = time.perf_counter()
        shap_values, _ = self.explainer.shap_values(
            input_tensor, ranked_outputs=1, nsamples=self.nsamples
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Output shape varies by SHAP version: older releases return a list
        # (one array per ranked class); shap 0.51 returns a single ndarray
        # (1, 3, 224, 224, n_ranked_outputs). Normalize to (3, 224, 224).
        sv = (
            np.asarray(shap_values[0]) if isinstance(shap_values, list) else np.asarray(shap_values)
        )
        sv = sv[0]  # drop the batch dim -> (3, 224, 224[, n_ranked_outputs])
        if sv.ndim == 4:  # trailing ranked-output axis (shap 0.51)
            sv = sv[..., 0]
        shap_map = np.abs(sv).sum(axis=0)  # (224, 224) — channel-aggregated
        peak = shap_map.max()
        if peak > 0:
            shap_map = shap_map / peak
        return shap_map, elapsed_ms
