"""P3 — Adversarial ε-curve via FGSM and PGD-k attacks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def _fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """ Change the input in the direction that increases the model’s loss as fast as possible.
     Because gradient vector gives direction of fastest increase in function"""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).detach()


def _pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.detach()


def probe_adversarial(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    epsilons: Iterable[float] = (0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255),
    attack: str = "pgd",
    pgd_steps: int = 10,
    pgd_alpha: float = 2 / 255,
) -> dict[str, Any]:
    """Accuracy under FGSM or PGD-k attacks at a list of ε values.

    Inputs are assumed to be in [0, 1]-ish (post-normalization is fine for this probe;
    we operate in the same input space the model sees and clip nothing).
    """
    model = model.to(device).eval()
    eps_list = list(epsilons)
    correct_by_eps = [0] * len(eps_list)
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        total += x.size(0)
        for i, eps in enumerate(eps_list):
            if eps == 0.0:
                preds = model(x).argmax(dim=1)
            elif attack == "fgsm":
                preds = model(_fgsm(model, x, y, eps)).argmax(dim=1)
            elif attack == "pgd":
                preds = model(_pgd(model, x, y, eps, pgd_alpha, pgd_steps)).argmax(dim=1)
            else:
                raise ValueError(f"unknown attack: {attack}")
            correct_by_eps[i] += int((preds == y).sum().item())
    accuracies = [c / max(1, total) for c in correct_by_eps]
    return {
        "attack": attack,
        "epsilons": eps_list,
        "accuracy_by_eps": accuracies,
        "robust_acc_at_8_255": (
            accuracies[eps_list.index(8 / 255)] if (8 / 255) in eps_list else None
        ),
    }
