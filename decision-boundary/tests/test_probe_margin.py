import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.margin import probe_margin


class _ConstantPredictor(nn.Module):
    """Always predict class 0 with logit gap of `gap` over all other classes."""

    def __init__(self, num_classes: int, gap: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gap = gap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        out = torch.zeros(b, self.num_classes)
        out[:, 0] = self.gap
        return out


def _toy_loader(num_classes: int = 5, n: int = 50) -> DataLoader:
    x = torch.randn(n, 1, 4, 4)
    y = torch.zeros(n, dtype=torch.long)  # all class 0
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_margin_for_constant_predictor_equals_gap() -> None:
    model = _ConstantPredictor(num_classes=5, gap=2.5)
    out = probe_margin(model, _toy_loader(), device="cpu")
    assert out["mean"] == 2.5
    assert out["median"] == 2.5
    assert out["min"] == 2.5
    assert "histogram" in out


def test_margin_can_be_negative_when_wrong() -> None:
    model = _ConstantPredictor(num_classes=5, gap=1.0)
    # Build a loader whose labels are all class 1 (wrong against the constant predictor)
    x = torch.randn(20, 1, 4, 4)
    y = torch.ones(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    out = probe_margin(model, loader, device="cpu")
    assert out["mean"] == -1.0  # logit[1]=0, max-other (class 0) = 1 → -1
