"""P2 — Mean L2 norm of ∇_x L(x, y)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader




def probe_input_grad(model: nn.Module, loader: DataLoader, device: str = "cuda") -> dict[str, Any]:
    """Average per-sample ‖∇_x cross_entropy(model(x), y)‖_2 over the loader."""
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    per_sample_norms: list[float] = []
    for x, y in loader:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        loss = criterion(model(x), y)
        grad = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]
        """If you set both create_graph=False and retain_graph=False inside torch.autograd.grad,
         PyTorch computes the standard first-order gradients while optimizing memory consumption.
         Here is exactly how these flags affect your model's runtime execution, memory behavior, and limitations:
         1. Memory Discarded (retain_graph=False) Setting retain_graph=False (which is the default PyTorch behavior)
          tells the engine to destroy the entire intermediate execution graph as soon as the backpropagation pass finishes.
          The Benefit: It immediately frees up the GPU/CPU memory used to store intermediate activations during the forward pass.
          The Limitation: You cannot call a second .backward() or another torch.autograd.grad function on any outputs stemming from that same forward pass.
           If you try, PyTorch will throw a runtime error stating that the graph has already been freed.
           2. No Higher-Order Derivatives (create_graph=False)Setting create_graph=False ensures that the gradient computation
            itself is not tracked or recorded.
            The Benefit: It reduces computational overhead and saves memory.
            The Limitation: You cannot compute higher-order derivatives (like double-derivatives, Hessians, or gradients of gradients).
             The resulting tensor will not have a backward history graph attached to it."""
        norms = grad.flatten(start_dim=1).norm(p=2, dim=1)
        per_sample_norms.extend(norms.detach().cpu().tolist())
    arr = np.asarray(per_sample_norms)
    counts, edges = np.histogram(arr, bins=50)
    return {
        "mean_grad_norm": float(arr.mean()),
        "median_grad_norm": float(np.median(arr)),
        "p90_grad_norm": float(np.percentile(arr, 90)),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
