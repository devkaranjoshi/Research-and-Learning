"""P4 — Mean distance from x along random directions before the prediction flips.

Inspired by Yang et al. 2020 "Boundary thickness and robustness".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_boundary_thickness(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_directions: int = 16,
    step: float = 1 / 255,
    max_steps: int = 64,
) -> dict[str, Any]:
    """Average distance from each sample along ``n_directions`` random unit directions
    before the model's prediction changes.

    Returns
    -------
    dict with ``mean_thickness``, ``median_thickness``, ``per_sample`` (per-sample mean).
    """
    model = model.to(device).eval()
    per_sample_thickness: list[float] = []
    for x, _ in loader:
        x = x.to(device)
        original_preds = model(x).argmax(dim=1)
        for sample_idx in range(x.size(0)):
            xs = x[sample_idx : sample_idx + 1]
            distances: list[float] = []
            for _ in range(n_directions):
                direction = torch.randn_like(xs)
                direction = direction / (direction.flatten().norm() + 1e-12)
                flipped_at = float("inf")
                for k in range(1, max_steps + 1):
                    x_perturbed = xs + k * step * direction
                    if model(x_perturbed).argmax(dim=1).item() != original_preds[sample_idx].item():
                        flipped_at = k * step
                        break
                distances.append(min(flipped_at, max_steps * step))
            per_sample_thickness.append(sum(distances) / len(distances))
    import numpy as np

    arr = np.asarray(per_sample_thickness)
    return {
        "mean_thickness": float(arr.mean()),
        "median_thickness": float(np.median(arr)),
        "p10_thickness": float(np.percentile(arr, 10)),
        "p90_thickness": float(np.percentile(arr, 90)),
        "per_sample": arr.tolist(),
    }
