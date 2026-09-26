import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.boundary_thickness import probe_boundary_thickness


class _LinearLogits(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            for c in range(num_classes):
                self.fc.weight[c, c % in_dim] = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_thicker_boundary_for_lower_logit_scale() -> None:
    torch.manual_seed(0)
    x = torch.rand(32, 4)
    y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    out_small = probe_boundary_thickness(
        _LinearLogits(4, 4, scale=0.1),
        loader,
        device="cpu",
        n_directions=8,
        step=0.1,
        max_steps=30,
    )
    out_large = probe_boundary_thickness(
        _LinearLogits(4, 4, scale=10.0),
        loader,
        device="cpu",
        n_directions=8,
        step=0.1,
        max_steps=30,
    )
    # Smaller logit scale ⇒ predictions easier to flip ⇒ thinner thickness.
    assert out_small["mean_thickness"] <= out_large["mean_thickness"] + 1e-6
