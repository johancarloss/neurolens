"""ResNet50 transfer-learning factory (Phase 2 — multi-architecture comparison).

Mirrors the public interface of ``vgg16.py`` (``build_*``, ``unfreeze_*``,
``get_target_layer_for_gradcam``) so the factory and the arch-agnostic runner
treat both architectures identically.

ResNet50 differs from VGG16 in two ways that matter here:
- its head is ``model.fc`` (not ``model.classifier``);
- global average pooling already produces a flat 2048-d vector before ``fc``,
  so the head is a plain ``Dropout -> Linear`` with no flatten or bottleneck.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(num_classes: int = 4, stage: int = 1, dropout: float = 0.5) -> nn.Module:
    """Build ResNet50 with a custom head and a 2-stage freeze schedule.

    Stage 1: backbone fully frozen; only ``model.fc`` (the head) trains.
    Stage 2: ``layer4`` (last bottleneck block) is unfrozen as well.

    Args:
        num_classes: output dimensionality. Default 4 (brain tumor classes).
        stage: 1 (head only) or 2 (head + layer4).
        dropout: dropout probability in the custom head.

    Returns:
        ``nn.Module`` ready for training.

    Raises:
        ValueError: if ``stage`` is not 1 or 2.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    # V2 weights use a newer torchvision training recipe (~+1.3% over V1).
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Stage 1: freeze the whole backbone (only the new head trains below).
    for p in model.parameters():
        p.requires_grad = False

    # Replace the 1000-class ImageNet head. ResNet50 pools to [N, 2048] before
    # ``fc``, so no Flatten is needed (unlike VGG16's 25088-wide head).
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if stage == 2:
        unfreeze_layer4(model)

    return model


def get_target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    """Return the recommended Grad-CAM target layer for ResNet50.

    ``layer4[-1]`` is the last bottleneck block; its 7x7 spatial output mirrors
    VGG16's last conv block, keeping the Phase 3 XAI comparison fair.
    """
    return model.layer4[-1]


def unfreeze_layer4(model: nn.Module) -> None:
    """Unfreeze layer4 in-place — used to transition stage 1 -> stage 2 mid-run."""
    for p in model.layer4.parameters():
        p.requires_grad = True
