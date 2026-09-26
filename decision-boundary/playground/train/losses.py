"""Loss helpers (label smoothing, mixup, cutmix)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_criterion(label_smoothing: float) -> nn.Module:
    """Cross-entropy with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (mixed_x, y_a, y_b, lam). alpha <= 0 returns identity."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(
    criterion: nn.Module, logits: torch.Tensor, y_a, y_b, lam: float
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
