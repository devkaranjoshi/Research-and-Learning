"""Two-hidden-layer MLP for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class MLP2(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))
