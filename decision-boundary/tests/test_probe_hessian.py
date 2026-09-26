import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.hessian import probe_hessian_top_eigenvalues


class _LinearLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_hessian_returns_k_eigenvalues_and_is_real() -> None:
    torch.manual_seed(0)
    model = _LinearLogits()
    x = torch.randn(64, 4)
    y = torch.randint(0, 4, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    out = probe_hessian_top_eigenvalues(model, loader, device="cpu", k=3, max_batches=2)
    assert len(out["top_eigenvalues"]) == 3
    for v in out["top_eigenvalues"]:
        assert isinstance(v, float)
    # The cross-entropy Hessian for a linear model is PSD, so eigenvalues >= 0.
    assert min(out["top_eigenvalues"]) >= -1e-3
