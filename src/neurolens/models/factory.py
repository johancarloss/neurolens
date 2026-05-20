"""Architecture dispatcher — single entry point used by trainers and XAI.

Phase 1 ships with VGG16 only. Phase 2 will add ResNet50 by registering
``"resnet50"`` in ``_MODEL_BUILDERS`` without touching any other module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from neurolens.models.vgg16 import (
    build_vgg16,
)
from neurolens.models.vgg16 import (
    get_target_layer_for_gradcam as _vgg16_gradcam_target,
)

# Registry of available architectures. To add a new arch (e.g., ResNet50 in
# Phase 2), import its ``build_*`` and ``get_target_layer_for_gradcam``,
# then add an entry below.
_MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "vgg16": build_vgg16,
}

_GRADCAM_TARGETS: dict[str, Callable[[nn.Module], nn.Module]] = {
    "vgg16": _vgg16_gradcam_target,
}


def build_model(
    arch: str,
    num_classes: int = 4,
    stage: int = 1,
    **kwargs: Any,
) -> nn.Module:
    """Build a model by architecture name.

    Args:
        arch: architecture identifier (e.g., ``"vgg16"``).
        num_classes: output dimensionality.
        stage: training stage (1=head only, 2=fine-tune late layers).
        **kwargs: arch-specific extras (e.g., ``dropout`` for VGG16).

    Returns:
        ``nn.Module`` ready for training.

    Raises:
        ValueError: if ``arch`` is not registered.
    """
    if arch not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown arch '{arch}'. Supported: {list(_MODEL_BUILDERS)}. "
            f"Phase 2 will add 'resnet50'."
        )
    return _MODEL_BUILDERS[arch](num_classes=num_classes, stage=stage, **kwargs)


def get_target_layer_for_gradcam(model: nn.Module, arch: str) -> nn.Module:
    """Return the appropriate Grad-CAM target layer for the given arch."""
    if arch not in _GRADCAM_TARGETS:
        raise ValueError(
            f"Grad-CAM target layer not defined for arch '{arch}'. "
            f"Supported: {list(_GRADCAM_TARGETS)}."
        )
    return _GRADCAM_TARGETS[arch](model)
