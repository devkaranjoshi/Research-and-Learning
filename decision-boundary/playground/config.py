"""Frozen run-configuration dataclass + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """All knobs for a single training run.

    Defaults match the v1 spec; any field not present in YAML keeps its default.
    """

    # --- core ---
    model: str = "lenet5"
    dataset: str = "mnist"
    epochs: int = 5
    seed: int = 0

    # --- optimization ---
    optimizer: str = "sgd_momentum"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"
    batch_size: int = 128
    sam_rho: float = 0.05
    gradient_clip: float | None = None

    # --- regularization ---
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    dropout: float = 0.0
    dropout_2d: float = 0.0
    data_augmentation: str = "none"
    stochastic_weight_averaging: bool = False
    swa_start_epoch: int = 0

    # --- architecture ---
    width_multiplier: float = 1.0
    depth: int = 20
    normalization: str = "batch_norm"
    activation: str = "relu"

    # --- data ---
    train_subset_fraction: float = 1.0
    label_noise_fraction: float = 0.0
    class_imbalance_factor: float = 1.0
    input_perturbation_sigma: float = 0.0

    # --- adversarial ---
    adv_training: str = "none"
    adv_eps: float = 8 / 255
    adv_alpha: float = 2 / 255

    # --- bookkeeping ---
    experiment_name: str = "adhoc"
    run_id: str = "run_000"
    extra: dict[str, Any] = field(default_factory=dict)


def _field_names() -> set[str]:
    return {f.name for f in fields(RunConfig)}


def _coerce_field_types(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce string values to int/float/bool for fields where the dataclass annotation
    is numeric.

    PyYAML's safe_load uses YAML 1.1 rules which leave values like ``5e-4`` as strings
    (no leading digit before the ``e``). This helper handles that case so authors can
    write ``weight_decay: 5e-4`` in configs without runtime TypeErrors downstream.
    """
    field_types = {f.name: f.type for f in fields(RunConfig)}
    coerced: dict[str, Any] = {}
    for key, value in data.items():
        ftype = field_types.get(key, "")
        ftype_str = ftype if isinstance(ftype, str) else getattr(ftype, "__name__", str(ftype))
        if isinstance(value, str) and "float" in ftype_str:
            try:
                coerced[key] = float(value)
                continue
            except ValueError:
                pass
        if isinstance(value, str) and "int" in ftype_str and "float" not in ftype_str:
            try:
                coerced[key] = int(value)
                continue
            except ValueError:
                pass
        coerced[key] = value
    return coerced


def load_run_config(path: Path) -> RunConfig:
    """Parse a YAML file into a RunConfig, rejecting unknown keys."""
    with Path(path).open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    known = _field_names()
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")
    return RunConfig(**_coerce_field_types(data))
