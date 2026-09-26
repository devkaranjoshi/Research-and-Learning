"""P7 - Loss along the linear interpolation between two parameter snapshots."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int) -> float:
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
        total += float(loss.item())
        n += x.size(0)
    return total / max(1, n)


def probe_linear_interp(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_points: int = 21,
    max_batches: int = 4,
) -> dict[str, Any]:
    """Loss along theta(alpha) = (1-alpha) * theta_A + alpha * theta_B for alpha in [0, 1]."""
    state_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    state_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}

    interp_model = copy.deepcopy(model_a).to(device).eval()
    alphas = np.linspace(0.0, 1.0, n_points)
    losses: list[float] = []
    for alpha in alphas:
        new_state = {
            k: (1.0 - alpha) * state_a[k].float() + alpha * state_b[k].float() for k in state_a
        }
        for k, v in new_state.items():
            new_state[k] = v.to(state_a[k].dtype)
        interp_model.load_state_dict(new_state)
        losses.append(_eval_loss(interp_model, loader, device, max_batches))
    return {"alphas": alphas.tolist(), "losses": losses}
