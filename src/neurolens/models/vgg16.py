"""VGG16 transfer-learning factory matching Wong et al. (2025, PLOS ONE)."""

from __future__ import annotations

from torch import nn
from torchvision.models import VGG16_Weights, vgg16

# VGG16 features output for 224x224 input is 7x7x512; flattened that's 25088.
_FLATTEN_FEATURES = 512 * 7 * 7

# In torchvision VGG16, the conv5 block (last 3 conv layers + ReLUs + MaxPool)
# starts at index 24 of the ``features`` sequential.
_CONV5_START_IDX = 24


def build_vgg16(num_classes: int = 4, stage: int = 1, dropout: float = 0.5) -> nn.Module:
    """Build VGG16 with Wong's custom head and a 2-stage freeze schedule.

    Wong's head: ``Flatten -> Linear(25088, 256) -> ReLU -> Dropout -> Linear(256, N)``
    (We drop the final softmax: ``CrossEntropyLoss`` applies it internally.)

    Stage 1: features module is fully frozen; only ``classifier`` trains.
    Stage 2: conv5 block (indices >= ``_CONV5_START_IDX``) is unfrozen as well.

    Args:
        num_classes: output dimensionality. Default 4 (brain tumor classes).
        stage: 1 (head only) or 2 (head + conv5).
        dropout: dropout probability in the custom head (Wong didn't specify;
            0.5 is the convention).

    Returns:
        ``nn.Module`` ready for training.

    Raises:
        ValueError: if ``stage`` is not 1 or 2.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Stage 1: freeze ALL feature layers (only classifier trainable below).
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace original 3-layer classifier with Wong's compact head.
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(_FLATTEN_FEATURES, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )

    if stage == 2:
        for idx in range(_CONV5_START_IDX, len(model.features)):
            for p in model.features[idx].parameters():
                p.requires_grad = True

    return model


def get_target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    """Return the recommended Grad-CAM target layer for VGG16.

    The last MaxPool in ``model.features`` keeps 7x7 spatial resolution,
    which gives meaningful saliency maps when upsampled to 224x224.
    """
    return model.features[-1]


def unfreeze_conv5(model: nn.Module) -> None:
    """Unfreeze conv5 in-place — used to transition stage 1 -> stage 2 mid-run."""
    for idx in range(_CONV5_START_IDX, len(model.features)):
        for p in model.features[idx].parameters():
            p.requires_grad = True
