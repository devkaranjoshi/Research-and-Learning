"""Parse sweep YAMLs into a list of (run_id, RunConfig) records."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from playground.config import RunConfig, _coerce_field_types, _field_names


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    config: RunConfig
    deep_probe: bool


def _validate_keys(d: dict[str, Any], where: str) -> None:
    unknown = set(d.keys()) - _field_names()
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def expand_sweep(path: Path) -> list[SweepRun]:
    """Return one SweepRun per cell in the Cartesian product of `sweep:` axes."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("sweep YAML must be a mapping")
    name = raw.get("experiment_name", path.stem)
    base = raw.get("base", {}) or {}
    sweep = raw.get("sweep", {}) or {}
    deep = raw.get("deep_probe_runs", []) or []
    _validate_keys(base, "base")
    _validate_keys(sweep, "sweep")

    base = _coerce_field_types(base)
    base_cfg = RunConfig(**base, experiment_name=name)

    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
    runs: list[SweepRun] = []
    for combo in itertools.product(*values):
        cell = dict(zip(keys, combo, strict=True))
        cell = _coerce_field_types(cell)
        run_id = f"{name}__" + "_".join(f"{k}={v}" for k, v in cell.items())
        cfg = replace(base_cfg, **cell, run_id=run_id)
        is_deep = any(all(cell.get(k) == v for k, v in dp.items()) for dp in deep)
        runs.append(SweepRun(run_id=run_id, config=cfg, deep_probe=is_deep))
    return runs
