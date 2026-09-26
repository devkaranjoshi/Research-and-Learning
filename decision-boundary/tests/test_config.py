from pathlib import Path

import pytest

from playground.config import RunConfig, load_run_config


def test_run_config_round_trips_yaml(tmp_path: Path) -> None:
    yaml_text = """
model: lenet5
dataset: mnist
optimizer: sgd_momentum
lr: 0.1
momentum: 0.9
weight_decay: 5e-4
batch_size: 128
epochs: 5
seed: 0
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)

    cfg = load_run_config(path)
    assert cfg.model == "lenet5"
    assert cfg.dataset == "mnist"
    assert cfg.batch_size == 128
    assert cfg.lr == pytest.approx(0.1)
    assert cfg.seed == 0


def test_run_config_is_frozen() -> None:
    cfg = RunConfig(model="lenet5", dataset="mnist")
    with pytest.raises((AttributeError, Exception)):
        cfg.model = "mlp2"  # frozen dataclass should reject this


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("model: lenet5\nnonsense_field: 42\n")
    with pytest.raises(ValueError, match="nonsense_field"):
        load_run_config(path)
