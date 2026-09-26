"""Deterministic seeding across torch, numpy, and python's random module."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the playground for reproducibility.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
