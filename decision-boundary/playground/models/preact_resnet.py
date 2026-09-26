"""Pre-activation ResNet for CIFAR-10.

Follows He et al. 2016 "Identity Mappings in Deep Residual Networks" closely enough
for the sharpness-research canon (Keskar 2017; Li et al. 2018).
"""

from __future__ import annotations

import torch
from torch import nn

from playground.models.small_cnn import _act, _norm


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.bn1 = _norm(normalization, in_planes)
        self.act1 = _act(activation)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = _norm(normalization, planes)
        self.act2 = _act(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(
                in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    """Configurable PreActResNet for 32x32 inputs (CIFAR-10).

    ``depth`` must satisfy (depth - 2) % 6 == 0. depth=20 -> n=3 blocks per stage,
    depth=56 -> n=9 blocks per stage.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 20,
        width_multiplier: float = 1.0,
        normalization: str = "batch_norm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must satisfy (depth - 2) % 6 == 0 (e.g. 20, 32, 44, 56)")
        n = (depth - 2) // 6
        c1 = max(8, int(16 * width_multiplier))
        c2 = max(16, int(32 * width_multiplier))
        c3 = max(32, int(64 * width_multiplier))

        self.in_planes = c1
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(
            c1, n, stride=1, normalization=normalization, activation=activation
        )
        self.layer2 = self._make_layer(
            c2, n, stride=2, normalization=normalization, activation=activation
        )
        self.layer3 = self._make_layer(
            c3, n, stride=2, normalization=normalization, activation=activation
        )
        self.bn_final = _norm(normalization, c3)
        self.act_final = _act(activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def _make_layer(
        self, planes: int, n: int, stride: int, normalization: str, activation: str
    ) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s, normalization, activation))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act_final(self.bn_final(out))
        out = self.pool(out).flatten(1)
        return self.fc(out)
