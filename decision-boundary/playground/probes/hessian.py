"""P5 - Top-k eigenvalues of the loss Hessian via Hessian-vector products + Lanczos.

We use ``scipy.sparse.linalg.eigsh`` on a LinearOperator that performs HVP through
``torch.autograd.grad``. This avoids any external loss-landscape dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import nn
from torch.utils.data import DataLoader


def _params_flat_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_flat_grads(model: nn.Module, flat_grad: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.grad = flat_grad[idx : idx + n].view_as(p)
        idx += n


def probe_hessian_top_eigenvalues(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    k: int = 5,
    max_batches: int = 8,
) -> dict[str, Any]:
    """Estimate top-k eigenvalues of the cross-entropy Hessian averaged over batches."""
    model = model.to(device).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n = _params_flat_size(model)

    # Pre-cache a small list of batches for stable HVP estimates.
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        cached.append((x.to(device), y.to(device)))

    def hvp(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(v_np).to(device=device, dtype=params[0].dtype)
        # Split v across params
        chunks: list[torch.Tensor] = []
        idx = 0
        for p in params:
            chunks.append(v[idx : idx + p.numel()].view_as(p))
            idx += p.numel()
        total_hvp = torch.zeros(n, device=device)
        for x, y in cached:
            loss = nn.functional.cross_entropy(model(x), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * c).sum() for g, c in zip(grads, chunks, strict=True))
            hvp_parts = torch.autograd.grad(dot, params, retain_graph=False)
            flat = torch.cat([h.flatten() for h in hvp_parts])
            total_hvp = total_hvp + flat.detach()
        return (total_hvp / len(cached)).cpu().numpy().astype(np.float64)

    linop = LinearOperator((n, n), matvec=hvp, dtype=np.float64)
    eigvals = eigsh(linop, k=min(k, n - 1), which="LA", return_eigenvectors=False, tol=1e-3)
    eigvals = sorted([float(v) for v in eigvals.tolist()], reverse=True)
    return {
        "top_eigenvalues": eigvals,
        "lambda_max": eigvals[0],
        "trace_estimate_sum_topk": float(sum(eigvals)),
    }
