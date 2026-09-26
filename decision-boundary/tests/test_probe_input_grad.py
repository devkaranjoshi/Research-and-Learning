import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.input_grad import probe_input_grad


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


def test_input_grad_norm_grows_with_logit_scale() -> None:
    x = torch.randn(16, 4)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    out_small = probe_input_grad(_LinearLogits(4, 4, scale=0.1), loader, device="cpu")
    out_large = probe_input_grad(_LinearLogits(4, 4, scale=10.0), loader, device="cpu")
    assert out_large["mean_grad_norm"] > out_small["mean_grad_norm"]
