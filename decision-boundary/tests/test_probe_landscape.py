import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.landscape import probe_loss_landscape_slice


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_landscape_returns_grid_centered_on_zero() -> None:
    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(16, 4)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    out = probe_loss_landscape_slice(
        model, loader, device="cpu", grid_size=5, span=0.5, max_batches=1
    )
    assert out["loss_grid"].shape == (5, 5)
    # The center cell corresponds to the model at θ = θ_0 → equals the unperturbed loss.
    center_loss = out["loss_grid"][2, 2]
    assert not np.isnan(center_loss)
