"""LeNet-5 (classic) for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
