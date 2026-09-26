import random

import numpy as np
import torch

from playground.seeding import seed_everything


def test_seed_everything_makes_torch_random_reproducible():
    seed_everything(42)
    a = torch.randn(3)
    seed_everything(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_seed_everything_makes_numpy_random_reproducible():
    seed_everything(42)
    a = np.random.randn(3)
    seed_everything(42)
    b = np.random.randn(3)
    assert (a == b).all()


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(42)
    a = [random.random() for _ in range(3)]
    seed_everything(42)
    b = [random.random() for _ in range(3)]
    assert a == b
