"""P6 — 2D filter-normalized loss-landscape slice (Li et al. 2018, NeurIPS).

For each weight tensor W, generate two random direction tensors D1, D2 with the same
shape. Per-filter renormalize each direction so it has the same per-filter Frobenius
norm as the corresponding W. Then evaluate
    L(W + a * D1 + b * D2)
on a (grid_size x grid_size) grid of (a, b) in [-span, span]^2.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _filter_normalized_direction(model: nn.Module) -> list[torch.Tensor]:
    dirs: list[torch.Tensor] = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if p.dim() == 1:  # bias/norm
            d.zero_()
        elif p.dim() == 2:  # linear: rows are "filters"
            d = d * (p.norm(dim=1, keepdim=True) / (d.norm(dim=1, keepdim=True) + 1e-12))
        elif p.dim() == 4:  # conv: out-channel is the filter
            p_norm = p.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            d_norm = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            d = d * (p_norm / d_norm)
        dirs.append(d)
    return dirs


@torch.no_grad()
def _eval_loss(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    total = 0.0
    n = 0
    for x, y in batches:
        loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
        total += float(loss.item())
        n += x.size(0)
    return total / max(1, n)


def probe_loss_landscape_slice(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    grid_size: int = 51,
    span: float = 1.0,
    max_batches: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a ``grid_size x grid_size`` 2D loss landscape slice around the model."""
    torch.manual_seed(seed)
    model = model.to(device).eval()
    base_state = copy.deepcopy(model.state_dict())
    params = list(model.parameters())
    d1 = _filter_normalized_direction(model)
    d2 = _filter_normalized_direction(model)

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        cached.append((x.to(device), y.to(device)))

    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)
    grid = np.empty((grid_size, grid_size), dtype=np.float64)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            model.load_state_dict(base_state)
            with torch.no_grad():
                for p, dd1, dd2 in zip(params, d1, d2, strict=True):
                    p.add_(a * dd1 + b * dd2)
            grid[i, j] = _eval_loss(model, cached)

    model.load_state_dict(base_state)
    return {
        "loss_grid": grid,
        "alphas": alphas,
        "betas": betas,
        "span": span,
    }
