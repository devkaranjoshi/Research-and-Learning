import json
from pathlib import Path

import yaml

from playground.runs_io import load_run, load_sweep


def _make_fake_run(run_dir: Path, batch_size: int, test_acc: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": "mlp2", "dataset": "mnist", "batch_size": batch_size, "seed": 0})
    )
    metrics = [
        {
            "epoch": 0,
            "train_loss": 1.0,
            "test_loss": 0.5,
            "test_accuracy": test_acc,
            "lr": 0.1,
        }
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
    (run_dir / "probes").mkdir()
    (run_dir / "probes" / "margin.json").write_text(json.dumps({"mean": 0.1, "median": 0.2}))


def test_load_run_reads_metrics_and_probes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_000"
    _make_fake_run(run_dir, batch_size=128, test_acc=0.95)
    run = load_run(run_dir)
    assert run.config["batch_size"] == 128
    assert len(run.metrics_df) == 1
    assert run.probes["margin"]["mean"] == 0.1


def test_load_sweep_returns_one_row_per_run(tmp_path: Path) -> None:
    _make_fake_run(tmp_path / "run_000", batch_size=32, test_acc=0.90)
    _make_fake_run(tmp_path / "run_001", batch_size=64, test_acc=0.92)
    df = load_sweep(tmp_path)
    assert len(df) == 2
    assert "batch_size" in df.columns
    assert "test_accuracy_final" in df.columns
    assert "margin_mean" in df.columns
