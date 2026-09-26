import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.adversarial import probe_adversarial


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _toy_loader() -> DataLoader:
    torch.manual_seed(0)
    x = torch.rand(32, 8)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_adversarial_accuracy_drops_monotonically_with_eps() -> None:
    model = _TinyClassifier()
    out = probe_adversarial(
        model,
        _toy_loader(),
        device="cpu",
        epsilons=[0.0, 0.01, 0.1, 0.5],
        attack="fgsm",
    )
    accs = out["accuracy_by_eps"]
    assert len(accs) == 4
    # Non-increasing
    for i in range(1, len(accs)):
        assert accs[i] <= accs[i - 1] + 1e-6


@pytest.mark.timeout(30)
def test_pgd_runs_and_returns_curve() -> None:
    model = _TinyClassifier()
    out = probe_adversarial(
        model,
        _toy_loader(),
        device="cpu",
        epsilons=[0.0, 0.05],
        attack="pgd",
        pgd_steps=5,
        pgd_alpha=0.01,
    )
    assert "accuracy_by_eps" in out
    assert len(out["accuracy_by_eps"]) == 2
