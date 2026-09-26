from pathlib import Path

from playground.sweep.config_loader import expand_sweep


def test_expand_sweep_produces_cartesian_product(tmp_path: Path) -> None:
    yaml_text = """
experiment_name: tiny_test
description: "tiny"
base:
  model: mlp2
  dataset: mnist
  epochs: 1
sweep:
  batch_size: [32, 64]
  seed: [0, 1, 2]
"""
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml_text)
    runs = expand_sweep(path)
    assert len(runs) == 6
    batches = sorted({r.config.batch_size for r in runs})
    seeds = sorted({r.config.seed for r in runs})
    assert batches == [32, 64]
    assert seeds == [0, 1, 2]
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == 6
    assert all(r.run_id.startswith("tiny_test") for r in runs)
