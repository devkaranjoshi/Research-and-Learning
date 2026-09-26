"""Learning-rate schedulers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from playground.config import RunConfig


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer)

    def get_lr(self):  # type: ignore[override]
        e = self.last_epoch
        if e < self.warmup_epochs:
            factor = (e + 1) / max(1, self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(optimizer: Optimizer, cfg: RunConfig):
    """Construct an LR scheduler. Returns None if cfg.lr_schedule == 'constant'."""
    if cfg.lr_schedule == "constant":
        return None
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.lr_schedule == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    if cfg.lr_schedule == "warmup_cosine":
        return WarmupCosineLR(
            optimizer, warmup_epochs=max(1, cfg.epochs // 10), total_epochs=cfg.epochs
        )
    raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule}")
