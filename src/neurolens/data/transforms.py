"""Image transforms for NeuroLens.

Replicates the augmentation choices of Wong et al. (2025, PLOS ONE) in
torchvision v2. The original paper used Keras ImageDataGenerator with
``shear_range=0.2``, ``zoom_range=0.2`` and ``horizontal_flip=True``.

We additionally apply ImageNet mean/std normalization because we use
ImageNet-pretrained VGG16 (Wong used rescale=1/255; our normalization is
more standard for transfer learning with ImageNet weights).
"""

from __future__ import annotations

import math

import torch
from torchvision.transforms import v2 as T  # noqa: N812 — community-standard alias

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(
    image_size: int = 224,
    shear_range: float = 0.2,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True,
) -> T.Compose:
    """Build the training-time augmentation pipeline.

    Mappings from Keras ``ImageDataGenerator`` to ``torchvision.transforms.v2``:

    * ``rescale=1/255``           -> handled by ``ToDtype(float32, scale=True)``
    * ``shear_range=0.2``         -> ``RandomAffine(shear=0.2 rad ~= 11.46°)``
    * ``zoom_range=0.2``          -> ``RandomResizedCrop(scale=(0.8, 1.2))``
    * ``horizontal_flip=True``    -> ``RandomHorizontalFlip(p=0.5)``

    Args:
        image_size: target square size for resize / crop (224 for VGG16).
        shear_range: shear range in radians (Keras convention; converted to
            degrees internally for torchvision).
        zoom_range: zoom fraction; results in scale=(1-zoom, 1+zoom) for the
            random crop.
        horizontal_flip: whether to include the random horizontal flip.

    Returns:
        A ``T.Compose`` that takes a PIL image and returns a normalized tensor
        of shape ``(3, image_size, image_size)``.
    """
    shear_degrees = shear_range * 180 / math.pi
    pipeline: list = [
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((image_size, image_size), antialias=True),
        T.RandomAffine(degrees=0, shear=shear_degrees),
        T.RandomResizedCrop(
            image_size,
            scale=(1.0 - zoom_range, 1.0 + zoom_range),
            ratio=(0.95, 1.05),
            antialias=True,
        ),
    ]
    if horizontal_flip:
        pipeline.append(T.RandomHorizontalFlip(p=0.5))
    pipeline += [
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(pipeline)


def eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for val/test (no augmentation, no randomness)."""
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.uint8, scale=True),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
