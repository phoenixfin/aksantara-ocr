"""From-scratch CNN baselines.

These matter for a dataset paper in a way pretrained backbones don't: they show
what the dataset alone teaches, without ImageNet features doing the work. The
depth ladder (2/3/4 blocks) doubles as a capacity ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, batch_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """Configurable VGG-style stack.

    Adaptive pooling before the classifier makes the model input-size agnostic,
    so the same architecture participates in the image-size ablation unchanged.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        depth: int = 3,
        base_width: int = 32,
        dropout: float = 0.5,
        batch_norm: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        blocks = []
        channels = in_channels
        for i in range(depth):
            out_ch = base_width * (2 ** i)
            blocks.append(_conv_block(channels, out_ch, batch_norm))
            channels = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class LeNet5(nn.Module):
    """The classic baseline. Included because character-recognition papers are
    expected to report it, and it anchors the low end of the capacity range."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
