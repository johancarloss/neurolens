"""Architecture dispatcher — single entry point used by trainers and XAI.

Supports VGG16 (Phase 1) and ResNet50 (Phase 2). Adding an architecture means
importing its ``build_*`` / ``get_target_layer_for_gradcam`` / ``unfreeze_*``
and registering them in the three dicts below — no other module changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from neurolens.models.resnet50 import (
    build_resnet50,
    unfreeze_layer4,
)
from neurolens.models.resnet50 import (
    get_target_layer_for_gradcam as _resnet50_gradcam_target,
)
from neurolens.models.vgg16 import (
    build_vgg16,
    unfreeze_conv5,
)
from neurolens.models.vgg16 import (
    get_target_layer_for_gradcam as _vgg16_gradcam_target,
)

# Registry of available architectures. To add a new arch, import its
# ``build_*`` / ``get_target_layer_for_gradcam`` / ``unfreeze_*`` and add one
# entry to each of the three dicts below.
_MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "vgg16": build_vgg16,
    "resnet50": build_resnet50,
}

_GRADCAM_TARGETS: dict[str, Callable[[nn.Module], nn.Module]] = {
    "vgg16": _vgg16_gradcam_target,
    "resnet50": _resnet50_gradcam_target,
}

# Maps arch -> the in-place unfreeze used to transition stage 1 -> stage 2.
# Lets the runner stay arch-agnostic instead of hardcoding ``unfreeze_conv5``.
_STAGE2_UNFREEZERS: dict[str, Callable[[nn.Module], None]] = {
    "vgg16": unfreeze_conv5,
    "resnet50": unfreeze_layer4,
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
        raise ValueError(f"Unknown arch '{arch}'. Supported: {list(_MODEL_BUILDERS)}.")
    return _MODEL_BUILDERS[arch](num_classes=num_classes, stage=stage, **kwargs)


def get_target_layer_for_gradcam(model: nn.Module, arch: str) -> nn.Module:
    """Return the appropriate Grad-CAM target layer for the given arch."""
    if arch not in _GRADCAM_TARGETS:
        raise ValueError(
            f"Grad-CAM target layer not defined for arch '{arch}'. "
            f"Supported: {list(_GRADCAM_TARGETS)}."
        )
    return _GRADCAM_TARGETS[arch](model)


def unfreeze_for_stage2(model: nn.Module, arch: str) -> None:
    """Unfreeze the arch-specific late block for stage-2 fine-tuning (in-place).

    VGG16 unfreezes conv5; ResNet50 unfreezes layer4. Keeping this behind the
    factory lets the training runner stay architecture-agnostic.

    Raises:
        ValueError: if no stage-2 unfreeze is registered for ``arch``.
    """
    if arch not in _STAGE2_UNFREEZERS:
        raise ValueError(
            f"No stage-2 unfreeze defined for arch '{arch}'. Supported: {list(_STAGE2_UNFREEZERS)}."
        )
    _STAGE2_UNFREEZERS[arch](model)
