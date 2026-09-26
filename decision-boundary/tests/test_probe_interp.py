import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.interp import probe_linear_interp


class _LinearLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_endpoints_match_each_models_loss() -> None:
    torch.manual_seed(0)
    model_a = _LinearLogits()
    model_b = _LinearLogits()
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(2.0)

    x = torch.randn(32, 4)
    y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    out = probe_linear_interp(model_a, model_b, loader, device="cpu", n_points=5)
    assert abs(out["alphas"][0]) < 1e-9
    assert abs(out["alphas"][-1] - 1.0) < 1e-9
    loss_a = out["losses"][0]
    loss_b = out["losses"][-1]
    assert loss_a != loss_b
