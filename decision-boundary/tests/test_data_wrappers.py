import torch
from torch.utils.data import TensorDataset

from playground.data.wrappers import (
    ClassImbalanceDataset,
    LabelNoiseDataset,
    SubsetFractionDataset,
)


def _toy_dataset(n: int = 100, num_classes: int = 10) -> TensorDataset:
    x = torch.randn(n, 1, 8, 8)
    y = torch.arange(n) % num_classes
    return TensorDataset(x, y)


def test_subset_fraction_returns_correct_size() -> None:
    ds = _toy_dataset(100)
    sub = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    assert len(sub) == 10


def test_subset_fraction_is_deterministic_for_same_seed() -> None:
    ds = _toy_dataset(100)
    a = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    b = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    assert [a[i][1].item() for i in range(len(a))] == [b[i][1].item() for i in range(len(b))]


def test_label_noise_flips_expected_fraction() -> None:
    ds = _toy_dataset(1000, num_classes=10)
    noisy = LabelNoiseDataset(ds, noise_fraction=0.3, num_classes=10, seed=0)
    original = torch.tensor([ds[i][1].item() for i in range(len(ds))])
    new = torch.tensor([noisy[i][1].item() for i in range(len(noisy))])
    changed = (original != new).float().mean().item()
    assert 0.20 < changed < 0.40  # noise should be ~30 %, with some slack


def test_class_imbalance_subset_skews_class_counts() -> None:
    ds = _toy_dataset(1000, num_classes=10)
    imbal = ClassImbalanceDataset(ds, imbalance_factor=10.0, num_classes=10, seed=0)
    counts = torch.zeros(10, dtype=torch.long)
    for i in range(len(imbal)):
        counts[imbal[i][1].item()] += 1
    assert counts.max().item() >= 10 * counts.min().item() - 5  # ~10x skew
