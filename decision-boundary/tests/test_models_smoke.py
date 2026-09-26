import pytest
import torch

from playground.models.registry import build_model


@pytest.mark.parametrize(
    "name, in_shape, num_classes",
    [
        ("mlp2", (1, 28, 28), 10),
        ("lenet5", (1, 28, 28), 10),
        ("small_cnn", (3, 32, 32), 10),
        ("preact_resnet20", (3, 32, 32), 10),
        ("preact_resnet56", (3, 32, 32), 10),
    ],
)
def test_model_forward_shape(name: str, in_shape: tuple[int, int, int], num_classes: int) -> None:
    model = build_model(name, num_classes=num_classes)
    x = torch.randn(4, *in_shape)
    y = model(x)
    assert y.shape == (4, num_classes)


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("totally_made_up", num_classes=10)


def test_width_multiplier_changes_param_count() -> None:
    base = build_model("small_cnn", num_classes=10, width_multiplier=1.0)
    wide = build_model("small_cnn", num_classes=10, width_multiplier=2.0)
    base_n = sum(p.numel() for p in base.parameters())
    wide_n = sum(p.numel() for p in wide.parameters())
    assert wide_n > base_n
