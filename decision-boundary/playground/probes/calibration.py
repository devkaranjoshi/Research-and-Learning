"""P8 - Expected Calibration Error + reliability bins."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_calibration(
    model: nn.Module, loader: DataLoader, device: str = "cuda", n_bins: int = 15
) -> dict[str, Any]:
    """Compute ECE and reliability-diagram bins."""
    model = model.to(device).eval()
    confs: list[float] = []
    correctness: list[int] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        confs.extend(conf.cpu().tolist())
        correctness.extend((pred == y).int().cpu().tolist())
    conf_arr = np.asarray(confs)
    corr_arr = np.asarray(correctness, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(conf_arr, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            bin_stats.append({"count": 0, "avg_confidence": None, "avg_accuracy": None})
            continue
        avg_conf = float(conf_arr[mask].mean())
        avg_acc = float(corr_arr[mask].mean())
        weight = mask.mean()
        ece += weight * abs(avg_conf - avg_acc)
        bin_stats.append(
            {"count": int(mask.sum()), "avg_confidence": avg_conf, "avg_accuracy": avg_acc}
        )
    return {
        "ece": float(ece),
        "bins": bins.tolist(),
        "bin_stats": bin_stats,
    }
