import json
from pathlib import Path

import pytest
import torch

from playground.config import RunConfig
from playground.models.registry import build_model
from playground.train.trainer import Trainer


def _make_toy_loaders(n_train: int = 256, n_test: int = 128, batch_size: int = 32):
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    x_train = torch.randn(n_train, 1, 28, 28)
    y_train = torch.randint(0, 10, (n_train,))
    x_test = torch.randn(n_test, 1, 28, 28)
    y_test = torch.randint(0, 10, (n_test,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size),
    )


@pytest.mark.timeout(60)
def test_trainer_writes_run_folder(tmp_path: Path) -> None:
    cfg = RunConfig(
        model="mlp2",
        dataset="mnist",
        optimizer="sgd_momentum",
        lr=0.01,
        epochs=2,
        batch_size=32,
        seed=0,
        lr_schedule="constant",
    )
    model = build_model(cfg.model)
    train_loader, test_loader = _make_toy_loaders()
    trainer = Trainer(model, cfg, run_dir=tmp_path / "run_000", device="cpu")
    trainer.fit(train_loader, test_loader)

    assert (tmp_path / "run_000" / "config.yaml").exists()
    assert (tmp_path / "run_000" / "metrics.jsonl").exists()
    assert (tmp_path / "run_000" / "checkpoints" / "last.pt").exists()
    # metrics.jsonl should have one line per epoch
    lines = (tmp_path / "run_000" / "metrics.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 2
    assert "train_loss" in parsed[0] and "test_accuracy" in parsed[0]
