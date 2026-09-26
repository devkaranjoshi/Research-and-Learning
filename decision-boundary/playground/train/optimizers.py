"""Optimizer builder. Includes Sharpness-Aware Minimization (Foret et al. 2021)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from playground.config import RunConfig


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around any base optimizer.

    Reference: Foret, Kleiner, Mobahi, Neyshabur, "Sharpness-Aware Minimization
    for Efficiently Improving Generalization" (ICLR 2021).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        base_optimizer_cls: type[Optimizer],
        rho: float = 0.05,
        **base_kwargs,
    ) -> None:
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):  # noqa: D401
        raise RuntimeError("SAM requires calling first_step() and second_step() explicitly")

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.zeros(1, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def build_optimizer(model: nn.Module, cfg: RunConfig) -> Optimizer:
    """Construct an optimizer from a RunConfig."""
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sam":
        return SAM(
            params,
            base_optimizer_cls=torch.optim.SGD,
            rho=cfg.sam_rho,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")
