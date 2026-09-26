"""Read sweep artifacts into tidy pandas dataframes for notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class Run:
    config: dict[str, Any]
    metrics_df: pd.DataFrame
    probes: dict[str, dict[str, Any]]
    run_dir: Path


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics_lines = (
        (run_dir / "metrics.jsonl").read_text().splitlines()
        if (run_dir / "metrics.jsonl").exists()
        else []
    )
    metrics = pd.DataFrame([json.loads(line) for line in metrics_lines if line.strip()])
    probes: dict[str, dict[str, Any]] = {}
    probe_dir = run_dir / "probes"
    if probe_dir.exists():
        for f in probe_dir.glob("*.json"):
            probes[f.stem] = json.loads(f.read_text())
    return Run(config=cfg, metrics_df=metrics, probes=probes, run_dir=run_dir)


def _summarize(run: Run) -> dict[str, Any]:
    row: dict[str, Any] = dict(run.config)
    if not run.metrics_df.empty:
        last = run.metrics_df.iloc[-1].to_dict()
        row["test_accuracy_final"] = last.get("test_accuracy")
        row["train_loss_final"] = last.get("train_loss")
        row["test_accuracy_best"] = run.metrics_df["test_accuracy"].max()
    for name, payload in run.probes.items():
        for k, v in payload.items():
            if isinstance(v, int | float | str | bool) or v is None:
                row[f"{name}_{k}"] = v
    row["run_dir"] = str(run.run_dir)
    return row


def load_sweep(experiment_dir: Path) -> pd.DataFrame:
    """One row per run; columns = config + final metrics + scalar probe values."""
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(Path(experiment_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        run = load_run(run_dir)
        rows.append(_summarize(run))
    return pd.DataFrame(rows)
