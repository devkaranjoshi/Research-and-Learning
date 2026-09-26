"""Single entry point for building any model in the zoo."""

from __future__ import annotations

from typing import Any

from torch import nn

from playground.models.lenet import LeNet5
from playground.models.mlp import MLP2
from playground.models.preact_resnet import PreActResNet
from playground.models.small_cnn import SmallCNN


def build_model(name: str, num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """Construct a model by name with optional architecture kwargs.

    Recognized kwargs (each model ignores those it doesn't accept):
        width_multiplier, normalization, activation, dropout, dropout_2d, depth
    """
    if name == "mlp2":
        return MLP2(num_classes=num_classes, dropout=kwargs.get("dropout", 0.0))
    if name == "lenet5":
        return LeNet5(num_classes=num_classes, dropout=kwargs.get("dropout", 0.0))
    if name == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            width_multiplier=kwargs.get("width_multiplier", 1.0),
            normalization=kwargs.get("normalization", "batch_norm"),
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.0),
            dropout_2d=kwargs.get("dropout_2d", 0.0),
        )
    if name in {"preact_resnet20", "preact_resnet56"}:
        depth = 20 if name == "preact_resnet20" else 56
        return PreActResNet(
            num_classes=num_classes,
            depth=kwargs.get("depth", depth),
            width_multiplier=kwargs.get("width_multiplier", 1.0),
            normalization=kwargs.get("normalization", "batch_norm"),
            activation=kwargs.get("activation", "relu"),
        )
    raise ValueError(f"unknown model: {name}")
