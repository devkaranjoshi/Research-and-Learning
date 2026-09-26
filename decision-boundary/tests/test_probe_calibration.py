import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.calibration import probe_calibration


class _Perfect(nn.Module):
    """Predicts class 0 with confidence ~0.99; ground truth is class 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        logits = torch.full((b, 5), -10.0)
        logits[:, 0] = 5.0
        return logits


def test_ece_is_low_when_confidence_matches_accuracy() -> None:
    model = _Perfect()
    x = torch.randn(64, 4)
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    out = probe_calibration(model, loader, device="cpu", n_bins=10)
    assert out["ece"] < 0.05
