"""Small 3-conv-block CIFAR-10 workhorse with width/normalization/activation dials."""

from __future__ import annotations

import torch
from torch import nn


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "group_norm":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    if kind == "layer_norm":
        # Channels-only layer norm via GroupNorm with 1 group.
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalization: {kind}")


def _act(kind: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
    }[kind]


class SmallCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        normalization: str = "batch_norm",
        activation: str = "relu",
        dropout: float = 0.0,
        dropout_2d: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = max(8, int(32 * width_multiplier))
        c2 = max(16, int(64 * width_multiplier))
        c3 = max(32, int(128 * width_multiplier))

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                _norm(normalization, cout),
                _act(activation),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                _norm(normalization, cout),
                _act(activation),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_2d),
            )

        self.features = nn.Sequential(block(3, c1), block(c1, c2), block(c2, c3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
