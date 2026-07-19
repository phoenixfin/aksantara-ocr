"""Augmentation policies — the axis of the augmentation ablation.

Each policy is a named point on a severity ladder so the paper can report
"accuracy vs. augmentation strength" as a clean monotonic table.

Handwritten script note: horizontal flips are deliberately absent from every
policy. Several Indonesian scripts contain character pairs that are mirror
images of each other, so a flip can turn one valid character into a different
valid character and silently corrupt the label.
"""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics — correct for pretrained backbones. Custom CNNs trained
# from scratch are unaffected by the specific constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAYSCALE_MEAN = (0.5,)
GRAYSCALE_STD = (0.5,)


def _normalize(grayscale: bool):
    if grayscale:
        return transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def build_transform(
    policy: str,
    image_size: int,
    train: bool,
    grayscale: bool = False,
):
    """Return a torchvision transform pipeline.

    Policies: ``none``, ``light``, ``medium``, ``heavy``.
    Evaluation always uses the deterministic ``none`` pipeline regardless of the
    requested policy — augmenting the test set would make results irreproducible.
    """
    base_end = [transforms.ToTensor(), _normalize(grayscale)]

    if not train or policy == "none":
        return transforms.Compose(
            [transforms.Resize((image_size, image_size))] + base_end
        )

    if policy == "light":
        aug = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ]
    elif policy == "medium":
        aug = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif policy == "heavy":
        aug = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {policy!r}")

    post = []
    if policy == "heavy":
        # Applied after ToTensor, so it must sit between tensor conversion and
        # normalization — hence the explicit ordering here.
        post = [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))]

    return transforms.Compose(aug + [transforms.ToTensor()] + post + [_normalize(grayscale)])
