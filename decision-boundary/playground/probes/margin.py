"""P1 — Margin distribution.

Margin for sample (x, y) is f(x)_y - max_{j != y} f(x)_j.
Positive ⇒ correct; magnitude ⇒ confidence-distance to the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_margin(model: nn.Module, loader: DataLoader, device: str = "cuda") -> dict[str, Any]:
    """Compute per-sample margins over the whole loader and return summary stats."""
    model = model.to(device).eval()
    margins: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        correct = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y.unsqueeze(1), float("-inf"))
        max_other, _ = masked.max(dim=1)
        margins.append((correct - max_other).cpu())
    flat = torch.cat(margins).numpy()
    counts, edges = np.histogram(flat, bins=50)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "frac_negative": float((flat < 0).mean()),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
