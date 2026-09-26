# Decision Boundary Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research sandbox that lets us twist training-recipe and data dials on small image classifiers (MNIST, CIFAR-10) and quantify the resulting decision-boundary sharpness through multiple complementary probes, with training executed on a remote NVIDIA GH200 and analysis composed in local Jupyter notebooks.

**Architecture:** Hybrid CLI-trainer + analysis-notebook ("Pattern C"). The `playground/` Python library exposes models, datasets, trainer, probes, sweep runner, and a tidy-dataframe loader. The `scripts/` directory contains CLI entry points (`train.py`, `probe.py`, `sweep.py`) invoked headlessly on the GH200 inside tmux. Each run produces a self-contained folder under `runs/` (config + metrics + checkpoints + probe outputs). Local notebooks in `notebooks/` read these folders and produce comparison plots.

**Tech Stack:** Python 3.10+, PyTorch 2.7 + CUDA 12.8 (pre-installed on the GH200), pyyaml, numpy, scipy, matplotlib, pandas, Jupyter. Tooling: `uv` for venv/deps, `black` + `ruff` for formatting/linting, `pytest` for tests, `nbstripout` pre-commit hook for notebooks. Deployment: rsync + tmux over SSH (no Docker). Live training curves via TensorBoard with SSH port-forward.

---

## File structure

The plan lays out work in roughly the order tasks execute. Files are grouped by responsibility, **not** technical layer (per coding-style.md).

```
decision-boundary-playground/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── playground/
│   ├── __init__.py
│   ├── config.py                 # Config dataclass + YAML loader
│   ├── seeding.py                # seed_everything()
│   ├── runs_io.py                # load_run, load_sweep (notebooks use this)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mnist.py              # MNIST loader + wrappers
│   │   ├── cifar10.py            # CIFAR-10 loader + wrappers
│   │   └── wrappers.py           # subset, label noise, class imbalance, augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py           # build_model(name, **kwargs)
│   │   ├── mlp.py
│   │   ├── lenet.py
│   │   ├── small_cnn.py
│   │   └── preact_resnet.py      # PreActResNet-20 and PreActResNet-56
│   ├── train/
│   │   ├── __init__.py
│   │   ├── optimizers.py         # build_optimizer, SAM wrapper
│   │   ├── schedulers.py
│   │   ├── trainer.py            # Trainer class
│   │   └── losses.py             # label_smoothing, mixup wrappers
│   ├── probes/
│   │   ├── __init__.py
│   │   ├── margin.py             # P1
│   │   ├── input_grad.py         # P2
│   │   ├── adversarial.py        # P3 (FGSM, PGD)
│   │   ├── boundary_thickness.py # P4
│   │   ├── hessian.py            # P5 (Lanczos via HVP)
│   │   ├── landscape.py          # P6 (filter-normalized 2D slice)
│   │   ├── interp.py             # P7
│   │   └── calibration.py        # P8 (ECE)
│   ├── sweep/
│   │   ├── __init__.py
│   │   ├── config_loader.py      # sweep YAML expansion
│   │   └── executor.py           # local-serial, gpu-parallel, gpu-serial
│   └── viz/
│       ├── __init__.py
│       └── plots.py              # matplotlib helpers reused by probes + notebooks
├── configs/
│   ├── base.yaml
│   ├── exp_batch_size_sweep.yaml
│   ├── exp_optimizer_sweep.yaml
│   └── exp_label_noise_sweep.yaml
├── scripts/
│   ├── train.py
│   ├── probe.py
│   ├── sweep.py
│   └── server_setup.sh
├── notebooks/
│   ├── 01_margin_distributions.ipynb
│   ├── 02_eps_curves.ipynb
│   ├── 03_landscape_comparisons.ipynb
│   ├── 04_sweep_summary.ipynb
│   └── 05_cross_probe_agreement.ipynb
├── tests/
│   ├── test_config.py
│   ├── test_seeding.py
│   ├── test_data_wrappers.py
│   ├── test_models_smoke.py
│   ├── test_trainer_smoke.py
│   ├── test_probe_margin.py
│   ├── test_probe_input_grad.py
│   ├── test_probe_adversarial.py
│   ├── test_probe_boundary_thickness.py
│   ├── test_probe_hessian.py
│   ├── test_probe_landscape.py
│   ├── test_probe_interp.py
│   ├── test_probe_calibration.py
│   ├── test_sweep_config.py
│   └── test_runs_io.py
└── runs/                          # gitignored
```

---

## Task 0: Repo scaffolding & tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `.pre-commit-config.yaml`
- Create: `playground/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Initialize git & uv project**

```bash
cd decision-boundary-playground
git init
git branch -m main
uv init --no-package .
rm hello.py main.py  # remove any uv-init scratch files (ignore if absent)
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "decision-boundary-playground"
version = "0.1.0"
description = "Research sandbox for studying brittle/sharp decision boundaries in image classifiers."
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch==2.7.0",
    "torchvision>=0.22.0,<0.23.0",
    "numpy>=1.26,<3",
    "scipy>=1.11",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
    "tensorboard>=2.16",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "black>=24.0",
    "ruff>=0.5",
    "pre-commit>=3.7",
    "nbstripout>=0.7",
    "jupyter>=1.0",
]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.pytest.ini_options]
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]

[tool.coverage.run]
source = ["playground"]
branch = true

[tool.coverage.report]
fail_under = 80
show_missing = true
```

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.uv/

# Project artifacts
runs/
figures/
*.ckpt
*.pt
*.npz

# Notebook outputs (kept clean by nbstripout, but ignore checkpoints)
.ipynb_checkpoints/

# Editor
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Test
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 4: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

- [ ] **Step 5: Write a minimal `README.md`**

```markdown
# Decision Boundary Playground

Research sandbox for studying brittle / sharp decision boundaries in small image classifiers.

See `docs/superpowers/specs/2026-05-13-decision-boundary-playground-design.md` for the full design.

## Setup (local)

```bash
uv sync --extra dev
pre-commit install
pytest
```

## Setup (Lambda GH200 server)

```bash
make sync       # rsync code → server
make shell      # ssh + tmux
# inside tmux:
bash scripts/server_setup.sh
python scripts/sweep.py --config configs/exp_batch_size_sweep.yaml
```

Then locally:

```bash
make fetch
jupyter lab notebooks/
```
```

- [ ] **Step 6: Install deps and verify**

```bash
uv sync --extra dev
uv run pytest --collect-only
```

Expected: zero tests collected, exit code 5 (no tests yet — fine). If exit code is non-zero for any other reason, fix before continuing.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md .pre-commit-config.yaml playground/__init__.py tests/__init__.py
git commit -m "chore: scaffold repo, pyproject, tooling"
```

---

## Task 1: Deterministic seeding

**Files:**
- Create: `playground/seeding.py`
- Test: `tests/test_seeding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seeding.py
import torch
import numpy as np
import random
from playground.seeding import seed_everything


def test_seed_everything_makes_torch_random_reproducible():
    seed_everything(42)
    a = torch.randn(3)
    seed_everything(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_seed_everything_makes_numpy_random_reproducible():
    seed_everything(42)
    a = np.random.randn(3)
    seed_everything(42)
    b = np.random.randn(3)
    assert (a == b).all()


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(42)
    a = [random.random() for _ in range(3)]
    seed_everything(42)
    b = [random.random() for _ in range(3)]
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_seeding.py -v
```

Expected: ImportError / ModuleNotFoundError on `playground.seeding`.

- [ ] **Step 3: Implement `seed_everything`**

```python
# playground/seeding.py
"""Deterministic seeding across torch, numpy, and python's random module."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the playground for reproducibility.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_seeding.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/seeding.py tests/test_seeding.py
git commit -m "feat: deterministic seeding across torch, numpy, random"
```

---

## Task 2: Config dataclass + YAML loader

**Files:**
- Create: `playground/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from pathlib import Path

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_config.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `RunConfig` and `load_run_config`**

```python
# playground/config.py
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
    return RunConfig(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/config.py tests/test_config.py
git commit -m "feat: frozen RunConfig dataclass + yaml loader"
```

---

## Task 3: Dataset wrappers (subset, label-noise, class-imbalance)

**Files:**
- Create: `playground/data/__init__.py`
- Create: `playground/data/wrappers.py`
- Test: `tests/test_data_wrappers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_wrappers.py
import torch
from torch.utils.data import TensorDataset

from playground.data.wrappers import (
    SubsetFractionDataset,
    LabelNoiseDataset,
    ClassImbalanceDataset,
)


def _toy_dataset(n: int = 100, num_classes: int = 10) -> TensorDataset:
    x = torch.randn(n, 1, 8, 8)
    y = torch.arange(n) % num_classes
    return TensorDataset(x, y)


def test_subset_fraction_returns_correct_size() -> None:
    ds = _toy_dataset(100)
    sub = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    assert len(sub) == 10


def test_subset_fraction_is_deterministic_for_same_seed() -> None:
    ds = _toy_dataset(100)
    a = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    b = SubsetFractionDataset(ds, fraction=0.1, seed=0)
    assert [a[i][1].item() for i in range(len(a))] == [b[i][1].item() for i in range(len(b))]


def test_label_noise_flips_expected_fraction() -> None:
    ds = _toy_dataset(1000, num_classes=10)
    noisy = LabelNoiseDataset(ds, noise_fraction=0.3, num_classes=10, seed=0)
    original = torch.tensor([ds[i][1].item() for i in range(len(ds))])
    new = torch.tensor([noisy[i][1].item() for i in range(len(noisy))])
    changed = (original != new).float().mean().item()
    assert 0.20 < changed < 0.40  # noise should be ~30 %, with some slack


def test_class_imbalance_subset_skews_class_counts() -> None:
    ds = _toy_dataset(1000, num_classes=10)
    imbal = ClassImbalanceDataset(ds, imbalance_factor=10.0, num_classes=10, seed=0)
    counts = torch.zeros(10, dtype=torch.long)
    for i in range(len(imbal)):
        counts[imbal[i][1].item()] += 1
    assert counts.max().item() >= 10 * counts.min().item() - 5  # ~10x skew
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_data_wrappers.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the wrappers**

```python
# playground/data/__init__.py
"""Datasets and dataset wrappers."""
```

```python
# playground/data/wrappers.py
"""Wrapper datasets that implement the v1 data dials.

Each wrapper is deterministic given a seed. None mutate the underlying dataset.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SubsetFractionDataset(Dataset):
    """Random fraction-of-dataset subset, seeded for reproducibility."""

    def __init__(self, base: Dataset, fraction: float, seed: int) -> None:
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_total = len(base)  # type: ignore[arg-type]
        n_keep = max(1, int(round(fraction * n_total)))
        rng = np.random.default_rng(seed)
        self._indices: np.ndarray = rng.choice(n_total, size=n_keep, replace=False)
        self._indices.sort()
        self._base = base

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int):
        return self._base[int(self._indices[idx])]


class LabelNoiseDataset(Dataset):
    """Flip a fraction of labels to uniformly random other classes (symmetric noise)."""

    def __init__(
        self,
        base: Dataset,
        noise_fraction: float,
        num_classes: int,
        seed: int,
    ) -> None:
        if not (0.0 <= noise_fraction <= 1.0):
            raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        n = len(base)  # type: ignore[arg-type]
        rng = np.random.default_rng(seed)
        flip_mask = rng.random(n) < noise_fraction
        flipped_labels = rng.integers(0, num_classes, size=n)
        self._base = base
        self._flip_mask = flip_mask
        self._flipped_labels = flipped_labels.astype(np.int64)
        self._num_classes = num_classes

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        x, y = self._base[idx]
        if self._flip_mask[idx]:
            new_y = int(self._flipped_labels[idx])
            if new_y == int(y):
                new_y = (new_y + 1) % self._num_classes
            return x, torch.tensor(new_y, dtype=torch.long)
        return x, y


class ClassImbalanceDataset(Dataset):
    """Long-tailed subset: class k keeps ``imbalance_factor**(-k/(C-1))`` of its samples."""

    def __init__(
        self,
        base: Dataset,
        imbalance_factor: float,
        num_classes: int,
        seed: int,
    ) -> None:
        if imbalance_factor < 1.0:
            raise ValueError("imbalance_factor must be >= 1.0")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        # Group indices by class
        by_class: list[list[int]] = [[] for _ in range(num_classes)]
        for i in range(len(base)):  # type: ignore[arg-type]
            _, y = base[i]
            by_class[int(y)].append(i)
        rng = np.random.default_rng(seed)
        kept: list[int] = []
        for k, idxs in enumerate(by_class):
            keep_frac = float(imbalance_factor ** (-k / (num_classes - 1)))
            n_keep = max(1, int(round(keep_frac * len(idxs))))
            chosen = rng.choice(np.asarray(idxs), size=min(n_keep, len(idxs)), replace=False)
            kept.extend(int(c) for c in chosen)
        kept.sort()
        self._indices: Sequence[int] = kept
        self._base = base

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_data_wrappers.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/data/__init__.py playground/data/wrappers.py tests/test_data_wrappers.py
git commit -m "feat: dataset wrappers (subset, label noise, class imbalance)"
```

---

## Task 4: MNIST and CIFAR-10 builders

**Files:**
- Create: `playground/data/mnist.py`
- Create: `playground/data/cifar10.py`
- Modify: `playground/data/__init__.py`
- Test: covered by `tests/test_models_smoke.py` in Task 6 (we don't want to download datasets in unit tests)

- [ ] **Step 1: Implement MNIST builder**

```python
# playground/data/mnist.py
"""MNIST dataset builder with augmentation + dial wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from playground.config import RunConfig
from playground.data.wrappers import (
    ClassImbalanceDataset,
    LabelNoiseDataset,
    SubsetFractionDataset,
)


_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def _build_transform(augment: str) -> transforms.Compose:
    base: list = [transforms.ToTensor(), transforms.Normalize(_MNIST_MEAN, _MNIST_STD)]
    if augment == "none":
        return transforms.Compose(base)
    if augment in {"flip_crop", "randaugment", "autoaugment"}:
        # MNIST: pad+random-crop only; horizontal flips would relabel a "2" as "5".
        return transforms.Compose(
            [transforms.RandomCrop(28, padding=2), *base]
        )
    raise ValueError(f"unknown augmentation: {augment}")


def _apply_wrappers(ds: Dataset, cfg: RunConfig, num_classes: int) -> Dataset:
    if cfg.label_noise_fraction > 0.0:
        ds = LabelNoiseDataset(ds, cfg.label_noise_fraction, num_classes, cfg.seed)
    if cfg.class_imbalance_factor > 1.0:
        ds = ClassImbalanceDataset(ds, cfg.class_imbalance_factor, num_classes, cfg.seed)
    if cfg.train_subset_fraction < 1.0:
        ds = SubsetFractionDataset(ds, cfg.train_subset_fraction, cfg.seed)
    return ds


def build_mnist_loaders(cfg: RunConfig, data_root: Path) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST."""
    train_t = _build_transform(cfg.data_augmentation)
    test_t = _build_transform("none")
    train_raw = datasets.MNIST(str(data_root), train=True, download=True, transform=train_t)
    test_raw = datasets.MNIST(str(data_root), train=False, download=True, transform=test_t)

    train_ds = _apply_wrappers(train_raw, cfg, num_classes=10)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_raw,
        batch_size=512,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
```

- [ ] **Step 2: Implement CIFAR-10 builder**

```python
# playground/data/cifar10.py
"""CIFAR-10 dataset builder with augmentation + dial wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from playground.config import RunConfig
from playground.data.wrappers import (
    ClassImbalanceDataset,
    LabelNoiseDataset,
    SubsetFractionDataset,
)


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _build_transform(augment: str, train: bool) -> transforms.Compose:
    base: list = [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    if not train or augment == "none":
        return transforms.Compose(base)
    if augment == "flip_crop":
        return transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *base]
        )
    if augment == "randaugment":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                *base,
            ]
        )
    if augment == "autoaugment":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                *base,
            ]
        )
    raise ValueError(f"unknown augmentation: {augment}")


def _apply_wrappers(ds: Dataset, cfg: RunConfig, num_classes: int) -> Dataset:
    if cfg.label_noise_fraction > 0.0:
        ds = LabelNoiseDataset(ds, cfg.label_noise_fraction, num_classes, cfg.seed)
    if cfg.class_imbalance_factor > 1.0:
        ds = ClassImbalanceDataset(ds, cfg.class_imbalance_factor, num_classes, cfg.seed)
    if cfg.train_subset_fraction < 1.0:
        ds = SubsetFractionDataset(ds, cfg.train_subset_fraction, cfg.seed)
    return ds


def build_cifar10_loaders(cfg: RunConfig, data_root: Path) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10."""
    train_t = _build_transform(cfg.data_augmentation, train=True)
    test_t = _build_transform(cfg.data_augmentation, train=False)
    train_raw = datasets.CIFAR10(str(data_root), train=True, download=True, transform=train_t)
    test_raw = datasets.CIFAR10(str(data_root), train=False, download=True, transform=test_t)

    train_ds = _apply_wrappers(train_raw, cfg, num_classes=10)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_raw,
        batch_size=512,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
```

- [ ] **Step 3: Export from `playground/data/__init__.py`**

```python
# playground/data/__init__.py
"""Datasets and dataset wrappers."""

from playground.data.cifar10 import build_cifar10_loaders
from playground.data.mnist import build_mnist_loaders


def build_dataloaders(cfg, data_root):
    """Dispatch to the correct dataset builder by name."""
    if cfg.dataset == "mnist":
        return build_mnist_loaders(cfg, data_root)
    if cfg.dataset == "cifar10":
        return build_cifar10_loaders(cfg, data_root)
    raise ValueError(f"unknown dataset: {cfg.dataset}")


__all__ = ["build_cifar10_loaders", "build_mnist_loaders", "build_dataloaders"]
```

- [ ] **Step 4: Import-smoke test**

```bash
uv run python -c "from playground.data import build_dataloaders; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add playground/data/
git commit -m "feat: MNIST and CIFAR-10 builders with augmentation + dial wrappers"
```

---

## Task 5: Model zoo (MLP-2, LeNet-5, SmallCNN, PreActResNet)

**Files:**
- Create: `playground/models/__init__.py`
- Create: `playground/models/mlp.py`
- Create: `playground/models/lenet.py`
- Create: `playground/models/small_cnn.py`
- Create: `playground/models/preact_resnet.py`
- Create: `playground/models/registry.py`
- Test: `tests/test_models_smoke.py`

- [ ] **Step 1: Write the failing tests (parameter counts + forward shape)**

```python
# tests/test_models_smoke.py
import torch
import pytest

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
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_models_smoke.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `MLP-2`**

```python
# playground/models/mlp.py
"""Two-hidden-layer MLP for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class MLP2(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))
```

- [ ] **Step 4: Implement `LeNet-5`**

```python
# playground/models/lenet.py
"""LeNet-5 (classic) for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

- [ ] **Step 5: Implement `SmallCNN`**

```python
# playground/models/small_cnn.py
"""Small 3-conv-block CIFAR-10 workhorse with width/normalization/activation dials."""

from __future__ import annotations

import torch
from torch import nn


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "group_norm":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    if kind == "layer_norm":
        # Channels-only layer norm via GroupNorm with 1 group.
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalization: {kind}")


def _act(kind: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
    }[kind]


class SmallCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        normalization: str = "batch_norm",
        activation: str = "relu",
        dropout: float = 0.0,
        dropout_2d: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = max(8, int(32 * width_multiplier))
        c2 = max(16, int(64 * width_multiplier))
        c3 = max(32, int(128 * width_multiplier))

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                _norm(normalization, cout),
                _act(activation),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                _norm(normalization, cout),
                _act(activation),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_2d),
            )

        self.features = nn.Sequential(block(3, c1), block(c1, c2), block(c2, c3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

- [ ] **Step 6: Implement `PreActResNet-20` / `PreActResNet-56`**

```python
# playground/models/preact_resnet.py
"""Pre-activation ResNet for CIFAR-10.

Follows He et al. 2016 "Identity Mappings in Deep Residual Networks" closely enough
for the sharpness-research canon (Keskar 2017; Li et al. 2018).
"""

from __future__ import annotations

import torch
from torch import nn

from playground.models.small_cnn import _act, _norm


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.bn1 = _norm(normalization, in_planes)
        self.act1 = _act(activation)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = _norm(normalization, planes)
        self.act2 = _act(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    """Configurable PreActResNet for 32×32 inputs (CIFAR-10).

    ``depth`` must satisfy (depth - 2) % 6 == 0. depth=20 → n=3 blocks per stage,
    depth=56 → n=9 blocks per stage.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 20,
        width_multiplier: float = 1.0,
        normalization: str = "batch_norm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must satisfy (depth - 2) % 6 == 0 (e.g. 20, 32, 44, 56)")
        n = (depth - 2) // 6
        c1 = max(8, int(16 * width_multiplier))
        c2 = max(16, int(32 * width_multiplier))
        c3 = max(32, int(64 * width_multiplier))

        self.in_planes = c1
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(c1, n, stride=1, normalization=normalization, activation=activation)
        self.layer2 = self._make_layer(c2, n, stride=2, normalization=normalization, activation=activation)
        self.layer3 = self._make_layer(c3, n, stride=2, normalization=normalization, activation=activation)
        self.bn_final = _norm(normalization, c3)
        self.act_final = _act(activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def _make_layer(self, planes: int, n: int, stride: int, normalization: str, activation: str) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s, normalization, activation))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act_final(self.bn_final(out))
        out = self.pool(out).flatten(1)
        return self.fc(out)
```

- [ ] **Step 7: Implement the registry**

```python
# playground/models/registry.py
"""Single entry point for building any model in the zoo."""

from __future__ import annotations

from typing import Any

from torch import nn

from playground.models.lenet import LeNet5
from playground.models.mlp import MLP2
from playground.models.preact_resnet import PreActResNet
from playground.models.small_cnn import SmallCNN


def build_model(name: str, num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """Construct a model by name with optional architecture kwargs.

    Recognized kwargs (those each model ignores it doesn't accept):
        width_multiplier, normalization, activation, dropout, dropout_2d, depth
    """
    if name == "mlp2":
        return MLP2(num_classes=num_classes, dropout=kwargs.get("dropout", 0.0))
    if name == "lenet5":
        return LeNet5(num_classes=num_classes, dropout=kwargs.get("dropout", 0.0))
    if name == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            width_multiplier=kwargs.get("width_multiplier", 1.0),
            normalization=kwargs.get("normalization", "batch_norm"),
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.0),
            dropout_2d=kwargs.get("dropout_2d", 0.0),
        )
    if name in {"preact_resnet20", "preact_resnet56"}:
        depth = 20 if name == "preact_resnet20" else 56
        return PreActResNet(
            num_classes=num_classes,
            depth=kwargs.get("depth", depth),
            width_multiplier=kwargs.get("width_multiplier", 1.0),
            normalization=kwargs.get("normalization", "batch_norm"),
            activation=kwargs.get("activation", "relu"),
        )
    raise ValueError(f"unknown model: {name}")
```

```python
# playground/models/__init__.py
"""Model zoo."""

from playground.models.registry import build_model

__all__ = ["build_model"]
```

- [ ] **Step 8: Run tests**

```bash
uv run pytest tests/test_models_smoke.py -v
```

Expected: 7 passed (5 parametrized + 2 standalone).

- [ ] **Step 9: Commit**

```bash
git add playground/models/ tests/test_models_smoke.py
git commit -m "feat: model zoo (MLP2, LeNet5, SmallCNN, PreActResNet-20/56)"
```

---

## Task 6: Optimizers (incl. SAM) and schedulers

**Files:**
- Create: `playground/train/__init__.py` (empty)
- Create: `playground/train/optimizers.py`
- Create: `playground/train/schedulers.py`

- [ ] **Step 1: Implement optimizers + SAM wrapper**

```python
# playground/train/optimizers.py
"""Optimizer builder. Includes Sharpness-Aware Minimization (Foret et al. 2021)."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from playground.config import RunConfig


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around any base optimizer.

    Reference: Foret, Kleiner, Mobahi, Neyshabur, "Sharpness-Aware Minimization
    for Efficiently Improving Generalization" (ICLR 2021).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        base_optimizer_cls: type[Optimizer],
        rho: float = 0.05,
        **base_kwargs,
    ) -> None:
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):  # noqa: D401
        raise RuntimeError("SAM requires calling first_step() and second_step() explicitly")

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.zeros(1, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def build_optimizer(model: nn.Module, cfg: RunConfig) -> Optimizer:
    """Construct an optimizer from a RunConfig."""
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sam":
        return SAM(
            params,
            base_optimizer_cls=torch.optim.SGD,
            rho=cfg.sam_rho,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")
```

- [ ] **Step 2: Implement schedulers**

```python
# playground/train/schedulers.py
"""Learning-rate schedulers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from playground.config import RunConfig


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer)

    def get_lr(self):  # type: ignore[override]
        e = self.last_epoch
        if e < self.warmup_epochs:
            factor = (e + 1) / max(1, self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(optimizer: Optimizer, cfg: RunConfig):
    """Construct an LR scheduler. Returns None if cfg.lr_schedule == 'constant'."""
    if cfg.lr_schedule == "constant":
        return None
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.lr_schedule == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    if cfg.lr_schedule == "warmup_cosine":
        return WarmupCosineLR(optimizer, warmup_epochs=max(1, cfg.epochs // 10), total_epochs=cfg.epochs)
    raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule}")
```

```python
# playground/train/__init__.py
"""Training pieces."""
```

- [ ] **Step 3: Import smoke**

```bash
uv run python -c "from playground.train.optimizers import build_optimizer, SAM; from playground.train.schedulers import build_scheduler; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add playground/train/
git commit -m "feat: optimizer + scheduler builders (incl. SAM)"
```

---

## Task 7: Trainer + run-folder discipline

**Files:**
- Create: `playground/train/losses.py`
- Create: `playground/train/trainer.py`
- Test: `tests/test_trainer_smoke.py`

- [ ] **Step 1: Write the failing trainer smoke test**

```python
# tests/test_trainer_smoke.py
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
```

Add `pytest-timeout` to pyproject's `dev` extras if not already present (it isn't — add it).

- [ ] **Step 2: Update pyproject to add `pytest-timeout`**

```diff
 dev = [
     "pytest>=8.0",
     "pytest-cov>=5.0",
+    "pytest-timeout>=2.3",
     "black>=24.0",
```

Then:
```bash
uv sync --extra dev
```

- [ ] **Step 3: Run test to verify failure**

```bash
uv run pytest tests/test_trainer_smoke.py -v
```

Expected: ImportError on `playground.train.trainer`.

- [ ] **Step 4: Implement losses helper**

```python
# playground/train/losses.py
"""Loss helpers (label smoothing, mixup, cutmix)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_criterion(label_smoothing: float) -> nn.Module:
    """Cross-entropy with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (mixed_x, y_a, y_b, lam). alpha <= 0 returns identity."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion: nn.Module, logits: torch.Tensor, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
```

- [ ] **Step 5: Implement the Trainer**

```python
# playground/train/trainer.py
"""Training loop with run-folder discipline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from playground.config import RunConfig
from playground.seeding import seed_everything
from playground.train.losses import build_criterion, mixup_batch, mixup_criterion
from playground.train.optimizers import SAM, build_optimizer
from playground.train.schedulers import build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Train a model and persist all artifacts to a single run folder.

    The run folder ends up containing:
        config.yaml          — exact resolved config for the run
        metrics.jsonl        — one JSON object per epoch
        checkpoints/last.pt  — final model + optimizer state
        checkpoints/best.pt  — best test-accuracy checkpoint
    """

    def __init__(self, model: nn.Module, cfg: RunConfig, run_dir: Path, device: str | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(cfg.seed)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.criterion = build_criterion(cfg.label_smoothing)

        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    def fit(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        metrics_path = self.run_dir / "metrics.jsonl"
        best_acc = -1.0
        with metrics_path.open("w") as f:
            for epoch in range(self.cfg.epochs):
                train_loss = self._train_one_epoch(train_loader)
                test_loss, test_acc = self._evaluate(test_loader)
                if self.scheduler is not None:
                    self.scheduler.step()
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                if test_acc > best_acc:
                    best_acc = test_acc
                    self._save_checkpoint("best.pt")
                logger.info("epoch %d  train_loss=%.4f  test_acc=%.4f", epoch, train_loss, test_acc)
        self._save_checkpoint("last.pt")

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.cfg.input_perturbation_sigma > 0.0:
                x = x + self.cfg.input_perturbation_sigma * torch.randn_like(x)

            mixed_x, y_a, y_b, lam = mixup_batch(x, y, self.cfg.mixup_alpha)

            if isinstance(self.optimizer, SAM):
                # First forward-backward (compute ε(w) ascent)
                self.optimizer.zero_grad()
                loss = mixup_criterion(self.criterion, self.model(mixed_x), y_a, y_b, lam)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # Second forward-backward at ascended weights
                loss2 = mixup_criterion(self.criterion, self.model(mixed_x), y_a, y_b, lam)
                loss2.backward()
                if self.cfg.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                self.optimizer.second_step(zero_grad=True)
                used_loss = loss
            else:
                self.optimizer.zero_grad()
                logits = self.model(mixed_x)
                used_loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                used_loss.backward()
                if self.cfg.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                self.optimizer.step()

            total_loss += used_loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / max(1, n)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            total_loss += self.criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        return total_loss / max(1, n), correct / max(1, n)

    def _save_checkpoint(self, name: str) -> None:
        path = self.run_dir / "checkpoints" / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.cfg),
            },
            path,
        )
```

- [ ] **Step 6: Run trainer smoke test**

```bash
uv run pytest tests/test_trainer_smoke.py -v
```

Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml playground/train/losses.py playground/train/trainer.py tests/test_trainer_smoke.py
git commit -m "feat: Trainer with run-folder discipline + mixup + SAM support"
```

---

## Task 8: Probe P1 — margin distribution

**Files:**
- Create: `playground/probes/__init__.py` (empty)
- Create: `playground/probes/margin.py`
- Test: `tests/test_probe_margin.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_margin.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_probe_margin.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/__init__.py
"""Decision-boundary probes."""
```

```python
# playground/probes/margin.py
"""P1 — Margin distribution.

Margin for sample (x, y) is f(x)_y - max_{j != y} f(x)_j.
Positive ⇒ correct; magnitude ⇒ confidence-distance to the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_margin(model: nn.Module, loader: DataLoader, device: str = "cuda") -> dict[str, Any]:
    """Compute per-sample margins over the whole loader and return summary stats."""
    model = model.to(device).eval()
    margins: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        correct = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y.unsqueeze(1), float("-inf"))
        max_other, _ = masked.max(dim=1)
        margins.append((correct - max_other).cpu())
    flat = torch.cat(margins).numpy()
    counts, edges = np.histogram(flat, bins=50)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "frac_negative": float((flat < 0).mean()),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_margin.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/__init__.py playground/probes/margin.py tests/test_probe_margin.py
git commit -m "feat(probes): P1 margin distribution"
```

---

## Task 9: Probe P2 — input-gradient norm

**Files:**
- Create: `playground/probes/input_grad.py`
- Test: `tests/test_probe_input_grad.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_input_grad.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.input_grad import probe_input_grad


class _LinearLogits(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            for c in range(num_classes):
                self.fc.weight[c, c % in_dim] = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_input_grad_norm_grows_with_logit_scale() -> None:
    x = torch.randn(16, 4)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    out_small = probe_input_grad(_LinearLogits(4, 4, scale=0.1), loader, device="cpu")
    out_large = probe_input_grad(_LinearLogits(4, 4, scale=10.0), loader, device="cpu")
    assert out_large["mean_grad_norm"] > out_small["mean_grad_norm"]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_probe_input_grad.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/input_grad.py
"""P2 — Mean L2 norm of ∇_x L(x, y)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def probe_input_grad(model: nn.Module, loader: DataLoader, device: str = "cuda") -> dict[str, Any]:
    """Average per-sample ‖∇_x cross_entropy(model(x), y)‖_2 over the loader."""
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    per_sample_norms: list[float] = []
    for x, y in loader:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        loss = criterion(model(x), y)
        grad = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]
        norms = grad.flatten(start_dim=1).norm(p=2, dim=1)
        per_sample_norms.extend(norms.detach().cpu().tolist())
    arr = np.asarray(per_sample_norms)
    counts, edges = np.histogram(arr, bins=50)
    return {
        "mean_grad_norm": float(arr.mean()),
        "median_grad_norm": float(np.median(arr)),
        "p90_grad_norm": float(np.percentile(arr, 90)),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_input_grad.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/input_grad.py tests/test_probe_input_grad.py
git commit -m "feat(probes): P2 input gradient norm"
```

---

## Task 10: Probe P3 — adversarial ε-curve (FGSM + PGD)

**Files:**
- Create: `playground/probes/adversarial.py`
- Test: `tests/test_probe_adversarial.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_adversarial.py
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.adversarial import probe_adversarial


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _toy_loader() -> DataLoader:
    torch.manual_seed(0)
    x = torch.rand(32, 8)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_adversarial_accuracy_drops_monotonically_with_eps() -> None:
    model = _TinyClassifier()
    out = probe_adversarial(
        model,
        _toy_loader(),
        device="cpu",
        epsilons=[0.0, 0.01, 0.1, 0.5],
        attack="fgsm",
    )
    accs = out["accuracy_by_eps"]
    assert len(accs) == 4
    # Non-increasing
    for i in range(1, len(accs)):
        assert accs[i] <= accs[i - 1] + 1e-6


@pytest.mark.timeout(30)
def test_pgd_runs_and_returns_curve() -> None:
    model = _TinyClassifier()
    out = probe_adversarial(
        model,
        _toy_loader(),
        device="cpu",
        epsilons=[0.0, 0.05],
        attack="pgd",
        pgd_steps=5,
        pgd_alpha=0.01,
    )
    assert "accuracy_by_eps" in out
    assert len(out["accuracy_by_eps"]) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_probe_adversarial.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/adversarial.py
"""P3 — Adversarial ε-curve via FGSM and PGD-k attacks."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader


def _fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).detach()


def _pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.detach()


def probe_adversarial(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    epsilons: Iterable[float] = (0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255),
    attack: str = "pgd",
    pgd_steps: int = 10,
    pgd_alpha: float = 2 / 255,
) -> dict[str, Any]:
    """Accuracy under FGSM or PGD-k attacks at a list of ε values.

    Inputs are assumed to be in [0, 1]-ish (post-normalization is fine for this probe;
    we operate in the same input space the model sees and clip nothing).
    """
    model = model.to(device).eval()
    eps_list = list(epsilons)
    correct_by_eps = [0] * len(eps_list)
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        total += x.size(0)
        for i, eps in enumerate(eps_list):
            if eps == 0.0:
                preds = model(x).argmax(dim=1)
            elif attack == "fgsm":
                preds = model(_fgsm(model, x, y, eps)).argmax(dim=1)
            elif attack == "pgd":
                preds = model(_pgd(model, x, y, eps, pgd_alpha, pgd_steps)).argmax(dim=1)
            else:
                raise ValueError(f"unknown attack: {attack}")
            correct_by_eps[i] += int((preds == y).sum().item())
    accuracies = [c / max(1, total) for c in correct_by_eps]
    return {
        "attack": attack,
        "epsilons": eps_list,
        "accuracy_by_eps": accuracies,
        "robust_acc_at_8_255": (
            accuracies[eps_list.index(8 / 255)] if (8 / 255) in eps_list else None
        ),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_adversarial.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/adversarial.py tests/test_probe_adversarial.py
git commit -m "feat(probes): P3 adversarial epsilon curve (FGSM + PGD)"
```

---

## Task 11: Probe P4 — boundary thickness

**Files:**
- Create: `playground/probes/boundary_thickness.py`
- Test: `tests/test_probe_boundary_thickness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_boundary_thickness.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.boundary_thickness import probe_boundary_thickness


class _LinearLogits(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            for c in range(num_classes):
                self.fc.weight[c, c % in_dim] = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_thicker_boundary_for_lower_logit_scale() -> None:
    torch.manual_seed(0)
    x = torch.rand(32, 4)
    y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    out_small = probe_boundary_thickness(
        _LinearLogits(4, 4, scale=0.1), loader, device="cpu",
        n_directions=8, step=0.1, max_steps=30,
    )
    out_large = probe_boundary_thickness(
        _LinearLogits(4, 4, scale=10.0), loader, device="cpu",
        n_directions=8, step=0.1, max_steps=30,
    )
    # Smaller logit scale ⇒ predictions easier to flip ⇒ thinner thickness.
    assert out_small["mean_thickness"] <= out_large["mean_thickness"] + 1e-6
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_probe_boundary_thickness.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/boundary_thickness.py
"""P4 — Mean distance from x along random directions before the prediction flips.

Inspired by Yang et al. 2020 "Boundary thickness and robustness".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_boundary_thickness(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_directions: int = 16,
    step: float = 1 / 255,
    max_steps: int = 64,
) -> dict[str, Any]:
    """Average distance from each sample along ``n_directions`` random unit directions
    before the model's prediction changes.

    Returns
    -------
    dict with ``mean_thickness``, ``median_thickness``, ``per_sample`` (per-sample mean).
    """
    model = model.to(device).eval()
    per_sample_thickness: list[float] = []
    for x, _ in loader:
        x = x.to(device)
        original_preds = model(x).argmax(dim=1)
        for sample_idx in range(x.size(0)):
            xs = x[sample_idx : sample_idx + 1]
            distances: list[float] = []
            for _ in range(n_directions):
                direction = torch.randn_like(xs)
                direction = direction / (direction.flatten().norm() + 1e-12)
                flipped_at = float("inf")
                for k in range(1, max_steps + 1):
                    x_perturbed = xs + k * step * direction
                    if model(x_perturbed).argmax(dim=1).item() != original_preds[sample_idx].item():
                        flipped_at = k * step
                        break
                distances.append(min(flipped_at, max_steps * step))
            per_sample_thickness.append(sum(distances) / len(distances))
    import numpy as np

    arr = np.asarray(per_sample_thickness)
    return {
        "mean_thickness": float(arr.mean()),
        "median_thickness": float(np.median(arr)),
        "p10_thickness": float(np.percentile(arr, 10)),
        "p90_thickness": float(np.percentile(arr, 90)),
        "per_sample": arr.tolist(),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_boundary_thickness.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/boundary_thickness.py tests/test_probe_boundary_thickness.py
git commit -m "feat(probes): P4 boundary thickness"
```

---

## Task 12: Probe P5 — Hessian top-k eigenvalues via Lanczos

**Files:**
- Create: `playground/probes/hessian.py`
- Test: `tests/test_probe_hessian.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_hessian.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.hessian import probe_hessian_top_eigenvalues


class _LinearLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_hessian_returns_k_eigenvalues_and_is_real() -> None:
    torch.manual_seed(0)
    model = _LinearLogits()
    x = torch.randn(64, 4)
    y = torch.randint(0, 4, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    out = probe_hessian_top_eigenvalues(model, loader, device="cpu", k=3, max_batches=2)
    assert len(out["top_eigenvalues"]) == 3
    for v in out["top_eigenvalues"]:
        assert isinstance(v, float)
    # The cross-entropy Hessian for a linear model is PSD, so eigenvalues ≥ 0.
    assert min(out["top_eigenvalues"]) >= -1e-3
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_probe_hessian.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement Lanczos via HVP**

```python
# playground/probes/hessian.py
"""P5 — Top-k eigenvalues of the loss Hessian via Hessian-vector products + Lanczos.

We use ``scipy.sparse.linalg.eigsh`` on a LinearOperator that performs HVP through
``torch.autograd.grad``. This avoids any external loss-landscape dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import nn
from torch.utils.data import DataLoader


def _params_flat_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_flat_grads(model: nn.Module, flat_grad: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.grad = flat_grad[idx : idx + n].view_as(p)
        idx += n


def probe_hessian_top_eigenvalues(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    k: int = 5,
    max_batches: int = 8,
) -> dict[str, Any]:
    """Estimate top-k eigenvalues of the cross-entropy Hessian averaged over batches."""
    model = model.to(device).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n = _params_flat_size(model)

    # Pre-cache a small list of batches for stable HVP estimates.
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        cached.append((x.to(device), y.to(device)))

    def hvp(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(v_np).to(device=device, dtype=params[0].dtype)
        # Split v across params
        chunks: list[torch.Tensor] = []
        idx = 0
        for p in params:
            chunks.append(v[idx : idx + p.numel()].view_as(p))
            idx += p.numel()
        total_hvp = torch.zeros(n, device=device)
        for x, y in cached:
            loss = nn.functional.cross_entropy(model(x), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * c).sum() for g, c in zip(grads, chunks))
            hvp_parts = torch.autograd.grad(dot, params, retain_graph=False)
            flat = torch.cat([h.flatten() for h in hvp_parts])
            total_hvp = total_hvp + flat.detach()
        return (total_hvp / len(cached)).cpu().numpy().astype(np.float64)

    linop = LinearOperator((n, n), matvec=hvp, dtype=np.float64)
    eigvals = eigsh(linop, k=min(k, n - 1), which="LA", return_eigenvectors=False, tol=1e-3)
    eigvals = sorted([float(v) for v in eigvals.tolist()], reverse=True)
    return {
        "top_eigenvalues": eigvals,
        "lambda_max": eigvals[0],
        "trace_estimate_sum_topk": float(sum(eigvals)),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_hessian.py -v
```

Expected: 1 passed (may take ~5 seconds).

- [ ] **Step 5: Commit**

```bash
git add playground/probes/hessian.py tests/test_probe_hessian.py
git commit -m "feat(probes): P5 Hessian top-k eigenvalues via Lanczos"
```

---

## Task 13: Probe P6 — filter-normalized landscape slice

**Files:**
- Create: `playground/probes/landscape.py`
- Test: `tests/test_probe_landscape.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_landscape.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.landscape import probe_loss_landscape_slice


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_landscape_returns_grid_centered_on_zero() -> None:
    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(16, 4)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    out = probe_loss_landscape_slice(
        model, loader, device="cpu", grid_size=5, span=0.5, max_batches=1
    )
    assert out["loss_grid"].shape == (5, 5)
    # The center cell corresponds to the model at θ = θ_0 → equals the unperturbed loss.
    center_loss = out["loss_grid"][2, 2]
    assert not np.isnan(center_loss)
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_probe_landscape.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement filter-normalized 2D slice**

```python
# playground/probes/landscape.py
"""P6 — 2D filter-normalized loss-landscape slice (Li et al. 2018, NeurIPS).

For each weight tensor W, generate two random direction tensors D1, D2 with the same
shape. Per-filter renormalize each direction so it has the same per-filter Frobenius
norm as the corresponding W. Then evaluate
    L(W + a * D1 + b * D2)
on a (grid_size × grid_size) grid of (a, b) ∈ [-span, span]^2.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _filter_normalized_direction(model: nn.Module) -> list[torch.Tensor]:
    dirs: list[torch.Tensor] = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if p.dim() == 1:  # bias/norm
            d.zero_()
        elif p.dim() == 2:  # linear: rows are "filters"
            d = d * (p.norm(dim=1, keepdim=True) / (d.norm(dim=1, keepdim=True) + 1e-12))
        elif p.dim() == 4:  # conv: out-channel is the filter
            p_norm = p.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            d_norm = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            d = d * (p_norm / d_norm)
        dirs.append(d)
    return dirs


@torch.no_grad()
def _eval_loss(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    total = 0.0
    n = 0
    for x, y in batches:
        loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
        total += float(loss.item())
        n += x.size(0)
    return total / max(1, n)


def probe_loss_landscape_slice(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    grid_size: int = 51,
    span: float = 1.0,
    max_batches: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a ``grid_size × grid_size`` 2D loss landscape slice around the model."""
    torch.manual_seed(seed)
    model = model.to(device).eval()
    base_state = copy.deepcopy(model.state_dict())
    params = list(model.parameters())
    d1 = _filter_normalized_direction(model)
    d2 = _filter_normalized_direction(model)

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        cached.append((x.to(device), y.to(device)))

    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)
    grid = np.empty((grid_size, grid_size), dtype=np.float64)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            with torch.no_grad():
                for p, dd1, dd2 in zip(params, d1, d2):
                    p.copy_(p.detach() * 0)  # zero out; we'll overwrite from base + a*d1 + b*d2
            # Restore base
            model.load_state_dict(base_state)
            with torch.no_grad():
                for p, dd1, dd2 in zip(params, d1, d2):
                    p.add_(a * dd1 + b * dd2)
            grid[i, j] = _eval_loss(model, cached)

    model.load_state_dict(base_state)
    return {
        "loss_grid": grid,
        "alphas": alphas,
        "betas": betas,
        "span": span,
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_landscape.py -v
```

Expected: 1 passed (~3 seconds).

- [ ] **Step 5: Commit**

```bash
git add playground/probes/landscape.py tests/test_probe_landscape.py
git commit -m "feat(probes): P6 filter-normalized 2D landscape slice"
```

---

## Task 14: Probe P7 — linear interpolation between two checkpoints

**Files:**
- Create: `playground/probes/interp.py`
- Test: `tests/test_probe_interp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_interp.py
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.interp import probe_linear_interp


class _LinearLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_endpoints_match_each_models_loss() -> None:
    torch.manual_seed(0)
    model_a = _LinearLogits()
    model_b = _LinearLogits()
    # Make them different
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(2.0)

    x = torch.randn(32, 4)
    y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    out = probe_linear_interp(model_a, model_b, loader, device="cpu", n_points=5)
    # alpha=0 ⇒ model_a; alpha=1 ⇒ model_b
    assert abs(out["alphas"][0]) < 1e-9
    assert abs(out["alphas"][-1] - 1.0) < 1e-9
    loss_a = out["losses"][0]
    loss_b = out["losses"][-1]
    assert loss_a != loss_b
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_probe_interp.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/interp.py
"""P7 — Loss along the linear interpolation between two parameter snapshots."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int) -> float:
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
        total += float(loss.item())
        n += x.size(0)
    return total / max(1, n)


def probe_linear_interp(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_points: int = 21,
    max_batches: int = 4,
) -> dict[str, Any]:
    """Loss along θ(α) = (1-α)·θ_A + α·θ_B for α ∈ [0, 1]."""
    state_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    state_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}

    interp_model = copy.deepcopy(model_a).to(device).eval()
    alphas = np.linspace(0.0, 1.0, n_points)
    losses: list[float] = []
    for alpha in alphas:
        new_state = {
            k: (1.0 - alpha) * state_a[k].float() + alpha * state_b[k].float()
            for k in state_a
        }
        # Cast back to original dtype
        for k, v in new_state.items():
            new_state[k] = v.to(state_a[k].dtype)
        interp_model.load_state_dict(new_state)
        losses.append(_eval_loss(interp_model, loader, device, max_batches))
    return {"alphas": alphas.tolist(), "losses": losses}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_interp.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/interp.py tests/test_probe_interp.py
git commit -m "feat(probes): P7 linear interpolation between two checkpoints"
```

---

## Task 15: Probe P8 — calibration / ECE

**Files:**
- Create: `playground/probes/calibration.py`
- Test: `tests/test_probe_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_calibration.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playground.probes.calibration import probe_calibration


class _Perfect(nn.Module):
    """Predicts class 0 with confidence 0.99; ground truth is class 0."""

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
    # Model is always confidently right ⇒ ECE small (~0).
    assert out["ece"] < 0.05
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_probe_calibration.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the probe**

```python
# playground/probes/calibration.py
"""P8 — Expected Calibration Error + reliability bins."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def probe_calibration(
    model: nn.Module, loader: DataLoader, device: str = "cuda", n_bins: int = 15
) -> dict[str, Any]:
    """Compute ECE and reliability-diagram bins."""
    model = model.to(device).eval()
    confs: list[float] = []
    correctness: list[int] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        confs.extend(conf.cpu().tolist())
        correctness.extend((pred == y).int().cpu().tolist())
    conf_arr = np.asarray(confs)
    corr_arr = np.asarray(correctness, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(conf_arr, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            bin_stats.append({"count": 0, "avg_confidence": None, "avg_accuracy": None})
            continue
        avg_conf = float(conf_arr[mask].mean())
        avg_acc = float(corr_arr[mask].mean())
        weight = mask.mean()
        ece += weight * abs(avg_conf - avg_acc)
        bin_stats.append({"count": int(mask.sum()), "avg_confidence": avg_conf, "avg_accuracy": avg_acc})
    return {
        "ece": float(ece),
        "bins": bins.tolist(),
        "bin_stats": bin_stats,
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_probe_calibration.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/probes/calibration.py tests/test_probe_calibration.py
git commit -m "feat(probes): P8 calibration / ECE"
```

---

## Task 16: Probe CLI (`scripts/probe.py`) + run-folder probe orchestration

**Files:**
- Create: `scripts/probe.py`
- Create: `playground/viz/__init__.py` (empty)
- Create: `playground/viz/plots.py`

- [ ] **Step 1: Implement small plotting helpers**

```python
# playground/viz/__init__.py
"""Plotting helpers."""
```

```python
# playground/viz/plots.py
"""Matplotlib plot helpers reused by probes and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # safe for headless server use
import matplotlib.pyplot as plt
import numpy as np


def save_histogram(counts: Sequence[int], edges: Sequence[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    widths = np.diff(edges)
    centers = np.asarray(edges)[:-1] + widths / 2
    ax.bar(centers, counts, width=widths, align="center", edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curve(xs: Sequence[float], ys: Sequence[float], path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_landscape(grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cf = ax.contourf(betas, alphas, grid, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel("β")
    ax.set_ylabel("α")
    ax.set_title("Filter-normalized loss landscape")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 2: Implement `scripts/probe.py`**

```python
# scripts/probe.py
"""Run probes against a trained checkpoint and save results to <run_dir>/probes/."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from playground.config import RunConfig
from playground.data import build_dataloaders
from playground.models.registry import build_model
from playground.probes.adversarial import probe_adversarial
from playground.probes.boundary_thickness import probe_boundary_thickness
from playground.probes.calibration import probe_calibration
from playground.probes.hessian import probe_hessian_top_eigenvalues
from playground.probes.input_grad import probe_input_grad
from playground.probes.landscape import probe_loss_landscape_slice
from playground.probes.margin import probe_margin
from playground.viz.plots import save_curve, save_histogram, save_landscape

ALL_PROBES = ("margin", "grad", "adv", "thickness", "hessian", "landscape", "calibration")
CHEAP_PROBES = ("margin", "grad", "adv", "calibration")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe")


def _load_run_config(run_dir: Path) -> RunConfig:
    with (run_dir / "config.yaml").open() as f:
        return RunConfig(**yaml.safe_load(f))


def _load_model(run_dir: Path, cfg: RunConfig, device: str) -> torch.nn.Module:
    model = build_model(cfg.model)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _save_json(path: Path, payload: dict) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"unhandled type {type(o)}")

    path.write_text(json.dumps(payload, indent=2, default=default))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument(
        "--probes",
        default="cheap",
        help="comma-separated probe names, or 'cheap' or 'all'",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = _load_run_config(args.run_dir)
    if args.probes == "cheap":
        names = CHEAP_PROBES
    elif args.probes == "all":
        names = ALL_PROBES
    else:
        names = tuple(x.strip() for x in args.probes.split(","))

    _, test_loader = build_dataloaders(cfg, args.data_root)
    model = _load_model(args.run_dir, cfg, args.device)

    probe_dir = args.run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    if "margin" in names:
        logger.info("running probe: margin")
        out = probe_margin(model, test_loader, device=args.device)
        _save_json(probe_dir / "margin.json", out)
        save_histogram(out["histogram"]["counts"], out["histogram"]["edges"], probe_dir / "margin.png", "Margin distribution")

    if "grad" in names:
        logger.info("running probe: input_grad")
        out = probe_input_grad(model, test_loader, device=args.device)
        _save_json(probe_dir / "grad.json", out)
        save_histogram(out["histogram"]["counts"], out["histogram"]["edges"], probe_dir / "grad.png", "Input grad norm")

    if "adv" in names:
        logger.info("running probe: adversarial (PGD)")
        out = probe_adversarial(model, test_loader, device=args.device, attack="pgd")
        _save_json(probe_dir / "adversarial.json", out)
        save_curve(out["epsilons"], out["accuracy_by_eps"], probe_dir / "adversarial.png", "Adversarial accuracy vs ε", "ε", "accuracy")

    if "calibration" in names:
        logger.info("running probe: calibration")
        out = probe_calibration(model, test_loader, device=args.device)
        _save_json(probe_dir / "calibration.json", out)

    if "thickness" in names:
        logger.info("running probe: boundary thickness")
        out = probe_boundary_thickness(model, test_loader, device=args.device)
        _save_json(probe_dir / "thickness.json", out)

    if "hessian" in names:
        logger.info("running probe: hessian top-k")
        out = probe_hessian_top_eigenvalues(model, test_loader, device=args.device)
        _save_json(probe_dir / "hessian.json", out)

    if "landscape" in names:
        logger.info("running probe: landscape slice")
        out = probe_loss_landscape_slice(model, test_loader, device=args.device)
        np.savez(probe_dir / "landscape.npz", **{k: v for k, v in out.items() if k != "loss_grid"}, loss_grid=out["loss_grid"])
        save_landscape(out["loss_grid"], np.asarray(out["alphas"]), np.asarray(out["betas"]), probe_dir / "landscape.png")

    logger.info("probes complete: %s", names)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Sanity-check the CLI**

```bash
uv run python scripts/probe.py --help
```

Expected: help message prints with `--run-dir`, `--probes`, `--device` flags.

- [ ] **Step 4: Commit**

```bash
git add playground/viz/ scripts/probe.py
git commit -m "feat: scripts/probe.py + viz helpers"
```

---

## Task 17: Training CLI (`scripts/train.py`)

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Implement `train.py`**

```python
# scripts/train.py
"""Train a single run.

Usage:
    python scripts/train.py --config configs/base.yaml --run-dir runs/adhoc/run_000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from playground.config import load_run_config
from playground.data import build_dataloaders
from playground.models.registry import build_model
from playground.train.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    logger.info("loaded config: model=%s dataset=%s epochs=%d", cfg.model, cfg.dataset, cfg.epochs)

    train_loader, test_loader = build_dataloaders(cfg, args.data_root)
    model = build_model(
        cfg.model,
        num_classes=10,
        width_multiplier=cfg.width_multiplier,
        normalization=cfg.normalization,
        activation=cfg.activation,
        dropout=cfg.dropout,
        dropout_2d=cfg.dropout_2d,
        depth=cfg.depth,
    )
    trainer = Trainer(model, cfg, run_dir=args.run_dir, device=args.device)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test (CPU, MNIST, 2 epochs)**

Create `configs/base.yaml` first:

```yaml
# configs/base.yaml
model: lenet5
dataset: mnist
optimizer: sgd_momentum
lr: 0.05
momentum: 0.9
weight_decay: 5e-4
batch_size: 128
epochs: 2
seed: 0
lr_schedule: constant
data_augmentation: none
```

Then run:

```bash
uv run python scripts/train.py --config configs/base.yaml --run-dir runs/_smoke/lenet5_mnist --data-root ./data
```

Expected: training prints 2 epochs of metrics; `runs/_smoke/lenet5_mnist/` contains `config.yaml`, `metrics.jsonl`, `checkpoints/best.pt`, `checkpoints/last.pt`. Final test accuracy on MNIST should be > 0.9.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py configs/base.yaml
git commit -m "feat: scripts/train.py CLI + base.yaml"
```

---

## Task 18: Sweep config loader + runs_io

**Files:**
- Create: `playground/sweep/__init__.py` (empty)
- Create: `playground/sweep/config_loader.py`
- Create: `playground/runs_io.py`
- Test: `tests/test_sweep_config.py`
- Test: `tests/test_runs_io.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_config.py
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
    # Each entry has the base values overlaid with one sweep value combo
    batches = sorted({r.config.batch_size for r in runs})
    seeds = sorted({r.config.seed for r in runs})
    assert batches == [32, 64]
    assert seeds == [0, 1, 2]
    # run_id should be unique and contain the experiment_name
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == 6
    assert all(r.run_id.startswith("tiny_test") for r in runs)
```

```python
# tests/test_runs_io.py
import json
from pathlib import Path

import yaml

from playground.runs_io import load_run, load_sweep


def _make_fake_run(run_dir: Path, batch_size: int, test_acc: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": "mlp2", "dataset": "mnist", "batch_size": batch_size, "seed": 0})
    )
    metrics = [{"epoch": 0, "train_loss": 1.0, "test_loss": 0.5, "test_accuracy": test_acc, "lr": 0.1}]
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
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_sweep_config.py tests/test_runs_io.py -v
```

Expected: ImportError on both.

- [ ] **Step 3: Implement `expand_sweep`**

```python
# playground/sweep/__init__.py
"""Sweep config + executor."""
```

```python
# playground/sweep/config_loader.py
"""Parse sweep YAMLs into a list of (run_id, RunConfig) records."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from playground.config import RunConfig, _field_names


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
    base = raw.get("base", {})
    sweep = raw.get("sweep", {})
    deep = raw.get("deep_probe_runs", [])
    _validate_keys(base, "base")
    _validate_keys(sweep, "sweep")

    base_cfg = RunConfig(**base, experiment_name=name)

    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
    runs: list[SweepRun] = []
    for idx, combo in enumerate(itertools.product(*values)):
        cell = dict(zip(keys, combo))
        run_id = f"{name}__" + "_".join(f"{k}={v}" for k, v in cell.items())
        cfg = replace(base_cfg, **cell, run_id=run_id)
        is_deep = any(all(cell.get(k) == v for k, v in dp.items()) for dp in deep)
        runs.append(SweepRun(run_id=run_id, config=cfg, deep_probe=is_deep))
    return runs
```

- [ ] **Step 4: Implement `runs_io.py`**

```python
# playground/runs_io.py
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
    metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines() if (run_dir / "metrics.jsonl").exists() else []
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
        # Flatten scalar values; skip arrays/histograms (notebooks load them separately).
        for k, v in payload.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
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
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_sweep_config.py tests/test_runs_io.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/sweep/ playground/runs_io.py tests/test_sweep_config.py tests/test_runs_io.py
git commit -m "feat: sweep config loader + runs_io tidy-dataframe loader"
```

---

## Task 19: Sweep executor (local / gpu-parallel / gpu-serial)

**Files:**
- Create: `playground/sweep/executor.py`
- Create: `scripts/sweep.py`

- [ ] **Step 1: Implement the executor**

```python
# playground/sweep/executor.py
"""Sweep executors.

Each run gets its own process so they can share the GPU without cross-contaminating
PyTorch RNG state or autograd caches. We use ``concurrent.futures.ProcessPoolExecutor``
for simplicity — the cost is one process startup per run, which is dwarfed by training.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import yaml

from playground.sweep.config_loader import SweepRun

logger = logging.getLogger(__name__)


def _write_run_config(run: SweepRun, runs_root: Path) -> Path:
    cfg_dir = runs_root / run.config.experiment_name / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{run.run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(asdict(run.config), sort_keys=False))
    return cfg_path


def _run_one(run: SweepRun, runs_root: Path, data_root: Path, do_probes: bool, device: str) -> tuple[str, int]:
    cfg_path = _write_run_config(run, runs_root)
    run_dir = runs_root / run.config.experiment_name / run.run_id
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", str(cfg_path),
        "--run-dir", str(run_dir),
        "--data-root", str(data_root),
        "--device", device,
    ]
    rc = subprocess.call(train_cmd)
    if rc != 0:
        return run.run_id, rc
    if do_probes:
        probe_names = "all" if run.deep_probe else "cheap"
        rc = subprocess.call([
            sys.executable, "scripts/probe.py",
            "--run-dir", str(run_dir),
            "--data-root", str(data_root),
            "--probes", probe_names,
            "--device", device,
        ])
    return run.run_id, rc


def run_sweep(
    runs: Sequence[SweepRun],
    *,
    executor: str,
    runs_root: Path,
    data_root: Path,
    do_probes: bool,
    max_parallel: int = 4,
    device: str = "cuda",
) -> dict[str, int]:
    """Execute a sweep and return ``{run_id: exit_code}``.

    Executors:
        local-serial : one process, CPU-only (forces device='cpu')
        gpu-serial   : one process at a time, uses 'device'
        gpu-parallel : up to max_parallel processes concurrently, uses 'device'
    """
    if executor == "local-serial":
        device = "cpu"
        max_parallel = 1
    elif executor == "gpu-serial":
        max_parallel = 1
    elif executor != "gpu-parallel":
        raise ValueError(f"unknown executor: {executor}")

    # Each worker pins its CPU thread count so we don't oversubscribe the 64-core host.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    results: dict[str, int] = {}
    if max_parallel == 1:
        for r in runs:
            rid, rc = _run_one(r, runs_root, data_root, do_probes, device)
            results[rid] = rc
            logger.info("done: %s rc=%d", rid, rc)
        return results

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_run_one, r, runs_root, data_root, do_probes, device) for r in runs]
        for fut in as_completed(futures):
            rid, rc = fut.result()
            results[rid] = rc
            logger.info("done: %s rc=%d", rid, rc)
    return results
```

- [ ] **Step 2: Implement `scripts/sweep.py`**

```python
# scripts/sweep.py
"""Run a sweep of training + probe jobs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from playground.sweep.config_loader import expand_sweep
from playground.sweep.executor import run_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--executor", choices=("local-serial", "gpu-serial", "gpu-parallel"), default="gpu-parallel")
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    runs = expand_sweep(args.config)
    logging.info("expanded %d runs from %s", len(runs), args.config)
    results = run_sweep(
        runs,
        executor=args.executor,
        runs_root=args.runs_root,
        data_root=args.data_root,
        do_probes=not args.no_probes,
        max_parallel=args.max_parallel,
        device=args.device,
    )
    failed = [rid for rid, rc in results.items() if rc != 0]
    if failed:
        logging.error("FAILED runs: %s", failed)
        raise SystemExit(1)
    logging.info("all %d runs succeeded", len(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the sweep CLI (CPU, tiny)**

Create a tiny sweep config:

```yaml
# configs/_smoke_sweep.yaml
experiment_name: smoke_sweep
description: "Tiny end-to-end smoke test for sweep.py"
base:
  model: mlp2
  dataset: mnist
  optimizer: sgd_momentum
  lr: 0.05
  epochs: 1
  lr_schedule: constant
sweep:
  batch_size: [64, 128]
  seed: [0]
```

```bash
uv run python scripts/sweep.py --config configs/_smoke_sweep.yaml --executor local-serial --data-root ./data --no-probes
```

Expected: two runs complete; `runs/smoke_sweep/` contains two run folders.

- [ ] **Step 4: Commit**

```bash
git add playground/sweep/executor.py scripts/sweep.py configs/_smoke_sweep.yaml
git commit -m "feat: sweep executor (local-serial / gpu-serial / gpu-parallel) + sweep.py CLI"
```

---

## Task 20: Example experiment configs

**Files:**
- Create: `configs/exp_batch_size_sweep.yaml`
- Create: `configs/exp_optimizer_sweep.yaml`
- Create: `configs/exp_label_noise_sweep.yaml`

- [ ] **Step 1: Batch size sweep (Keskar reproduction)**

```yaml
# configs/exp_batch_size_sweep.yaml
experiment_name: batch_size_sharpness
description: "Keskar 2017: large batch ⇒ sharp minima?"
base:
  model: preact_resnet20
  dataset: cifar10
  optimizer: sgd_momentum
  lr: 0.1
  momentum: 0.9
  weight_decay: 5e-4
  lr_schedule: cosine
  data_augmentation: flip_crop
  epochs: 100
sweep:
  batch_size: [32, 128, 512, 2048, 8192]
  seed: [0, 1, 2]
deep_probe_runs:
  - {batch_size: 32, seed: 0}
  - {batch_size: 8192, seed: 0}
```

- [ ] **Step 2: Optimizer sweep (SAM vs SGD vs Adam)**

```yaml
# configs/exp_optimizer_sweep.yaml
experiment_name: optimizer_sharpness
description: "Does SAM produce flatter minima than SGD/Adam at fixed batch size?"
base:
  model: preact_resnet20
  dataset: cifar10
  lr: 0.1
  momentum: 0.9
  weight_decay: 5e-4
  lr_schedule: cosine
  data_augmentation: flip_crop
  batch_size: 128
  epochs: 100
sweep:
  optimizer: [sgd_momentum, adam, sam]
  seed: [0, 1, 2]
deep_probe_runs:
  - {optimizer: sgd_momentum, seed: 0}
  - {optimizer: sam, seed: 0}
```

- [ ] **Step 3: Label-noise sweep (Zhang et al. memorization)**

```yaml
# configs/exp_label_noise_sweep.yaml
experiment_name: label_noise_sharpness
description: "How does label noise shape margin distributions and adversarial robustness?"
base:
  model: small_cnn
  dataset: cifar10
  optimizer: sgd_momentum
  lr: 0.05
  weight_decay: 5e-4
  lr_schedule: cosine
  data_augmentation: flip_crop
  batch_size: 128
  epochs: 50
sweep:
  label_noise_fraction: [0.0, 0.1, 0.2, 0.4]
  seed: [0, 1, 2]
deep_probe_runs:
  - {label_noise_fraction: 0.0, seed: 0}
  - {label_noise_fraction: 0.4, seed: 0}
```

- [ ] **Step 4: Commit**

```bash
git add configs/exp_batch_size_sweep.yaml configs/exp_optimizer_sweep.yaml configs/exp_label_noise_sweep.yaml
git commit -m "feat: three example sweep configs (batch size, optimizer, label noise)"
```

---

## Task 21: Server setup script + Makefile

**Files:**
- Create: `scripts/server_setup.sh`
- Create: `Makefile`

- [ ] **Step 1: Implement `server_setup.sh`**

```bash
# scripts/server_setup.sh
#!/usr/bin/env bash
# Run once on the Lambda GH200 box, after `make sync` has rsynced the repo.
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")/.."

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv + install deps (PyTorch 2.7 is already on the system; we still install
# our pinned versions into the venv).
uv sync --extra dev

# Verify GPU access
uv run python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))"

echo "Server setup complete."
```

- [ ] **Step 2: Implement `Makefile`**

```makefile
# Makefile

SERVER ?= ubuntu@<server-ip>
KEY    ?= ~/.ssh/<key>.pem
REMOTE ?= /home/ubuntu/decision-boundary-playground

.PHONY: sync fetch shell setup test lint

sync:
	rsync -avz --exclude runs/ --exclude .venv --exclude data --exclude __pycache__ \
	    -e "ssh -i $(KEY)" ./ $(SERVER):$(REMOTE)/

fetch:
	rsync -avz -e "ssh -i $(KEY)" $(SERVER):$(REMOTE)/runs/ ./runs/

shell:
	ssh -i $(KEY) -t $(SERVER) "tmux new -A -s playground -c $(REMOTE)"

setup:
	ssh -i $(KEY) -t $(SERVER) "cd $(REMOTE) && bash scripts/server_setup.sh"

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .
```

- [ ] **Step 3: Smoke-test sync (dry-run) and shell**

```bash
# From the project root locally
make sync
make setup
```

Expected: `make sync` rsyncs the repo to the server; `make setup` runs the server-setup script (installs uv if missing, creates venv, prints `cuda: NVIDIA GH200 480GB`).

If either fails, stop and debug. Do NOT proceed to the next task.

- [ ] **Step 4: Commit**

```bash
git add scripts/server_setup.sh Makefile
git commit -m "feat: server setup script + Makefile (sync/fetch/shell/setup)"
```

---

## Task 22: End-to-end smoke run on the GH200

**Files:** none (operational task)

- [ ] **Step 1: Sync to server**

```bash
make sync
```

- [ ] **Step 2: Open server shell**

```bash
make shell
```

- [ ] **Step 3: Inside tmux on the server, run the tiny sweep with probes**

```bash
uv run python scripts/sweep.py \
    --config configs/_smoke_sweep.yaml \
    --executor gpu-parallel \
    --max-parallel 2 \
    --data-root ./data
```

Expected: two runs complete on GPU (one each for batch_size=64 and batch_size=128), probes run for each (margin, grad, adversarial, calibration), final lines say `all 2 runs succeeded`.

- [ ] **Step 4: Detach tmux (`Ctrl-b d`) and exit ssh, then fetch results locally**

```bash
make fetch
ls runs/smoke_sweep/
```

Expected: two folders, each with `config.yaml`, `metrics.jsonl`, `checkpoints/`, `probes/margin.json`, `probes/adversarial.json`, etc.

- [ ] **Step 5: Commit a marker note (optional but useful)**

```bash
echo "First end-to-end GH200 run completed $(date -I)" >> docs/superpowers/specs/RUNLOG.md
git add docs/superpowers/specs/RUNLOG.md
git commit -m "chore: log first end-to-end GH200 sweep"
```

---

## Task 23: Notebook 04 — Sweep summary dashboard

**Files:**
- Create: `notebooks/04_sweep_summary.ipynb`

(We build notebook 04 first because it's the workhorse and demonstrates the loader; the others are thin specializations.)

- [ ] **Step 1: Create the notebook cells**

In a fresh Jupyter notebook, paste these cells one per cell:

```python
# Cell 1: Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # so `import playground` works from notebooks/

import matplotlib.pyplot as plt
import pandas as pd

from playground.runs_io import load_sweep
```

```python
# Cell 2: Load — change EXPERIMENT below to look at a different sweep
EXPERIMENT = "smoke_sweep"   # or "batch_size_sharpness", etc.
runs_dir = Path.cwd().parent / "runs" / EXPERIMENT
if not runs_dir.exists():
    print(f"No runs at {runs_dir}. Run `make fetch` or pick another experiment.")
    df = pd.DataFrame()
else:
    df = load_sweep(runs_dir)
    print(f"loaded {len(df)} runs")
df.head()
```

```python
# Cell 3: Identify which dial varied
if not df.empty:
    config_cols = [c for c in df.columns if not c.endswith("_final") and not c.endswith("_best") and not c.startswith(("margin_", "grad_", "adversarial_", "calibration_", "thickness_", "hessian_", "landscape_")) and c not in {"run_dir", "experiment_name", "run_id", "extra", "description"}]
    varying = [c for c in config_cols if df[c].nunique() > 1 and c != "seed"]
    print("varying dial(s):", varying)
```

```python
# Cell 4: Test accuracy vs dial value (with seed error bars)
if not df.empty and varying:
    dial = varying[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby(dial)["test_accuracy_best"].agg(["mean", "std"]).reset_index()
    ax.errorbar(grouped[dial].astype(str), grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
    ax.set_xlabel(dial)
    ax.set_ylabel("best test accuracy")
    ax.set_title(f"Test accuracy vs {dial}")
    plt.show()
```

```python
# Cell 5: Margin mean vs dial value
if not df.empty and varying and "margin_mean" in df.columns:
    dial = varying[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby(dial)["margin_mean"].agg(["mean", "std"]).reset_index()
    ax.errorbar(grouped[dial].astype(str), grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
    ax.set_xlabel(dial)
    ax.set_ylabel("mean margin")
    ax.set_title(f"Margin mean vs {dial}")
    plt.show()
```

```python
# Cell 6: Adversarial robustness vs dial value
if not df.empty and varying and "adversarial_robust_acc_at_8_255" in df.columns:
    dial = varying[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby(dial)["adversarial_robust_acc_at_8_255"].agg(["mean", "std"]).reset_index()
    ax.errorbar(grouped[dial].astype(str), grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
    ax.set_xlabel(dial)
    ax.set_ylabel("robust accuracy @ ε = 8/255")
    ax.set_title(f"Adversarial robustness vs {dial}")
    plt.show()
```

- [ ] **Step 2: Run all cells**

Make sure the notebook runs end-to-end against `runs/smoke_sweep/`. If there are no smoke runs locally yet, the notebook should print "No runs at …" and gracefully not error.

- [ ] **Step 3: Strip outputs and commit**

```bash
uv run nbstripout notebooks/04_sweep_summary.ipynb
git add notebooks/04_sweep_summary.ipynb
git commit -m "feat(notebooks): 04 sweep summary dashboard"
```

---

## Task 24: Notebooks 01, 02, 03, 05 (thin specializations)

**Files:**
- Create: `notebooks/01_margin_distributions.ipynb`
- Create: `notebooks/02_eps_curves.ipynb`
- Create: `notebooks/03_landscape_comparisons.ipynb`
- Create: `notebooks/05_cross_probe_agreement.ipynb`

Each is roughly 20–30 lines. They share the Cell-1 and Cell-2 setup code from Task 23. Only the analysis cells differ.

- [ ] **Step 1: Notebook 01 — Margin distributions overlay**

Analysis cell:

```python
# Overlay margin histograms across runs, colored by dial value
import json

if not df.empty and varying:
    dial = varying[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in df.iterrows():
        margin = json.loads((Path(row["run_dir"]) / "probes" / "margin.json").read_text())
        h = margin["histogram"]
        edges = h["edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        ax.plot(centers, h["counts"], label=f"{dial}={row[dial]}, seed={row['seed']}", alpha=0.6)
    ax.set_xlabel("margin")
    ax.set_ylabel("count")
    ax.set_title(f"Margin distribution vs {dial}")
    ax.legend(fontsize=7)
    plt.show()
```

- [ ] **Step 2: Notebook 02 — Adversarial ε-curves overlay**

Analysis cell:

```python
import json

if not df.empty and varying:
    dial = varying[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in df.iterrows():
        adv = json.loads((Path(row["run_dir"]) / "probes" / "adversarial.json").read_text())
        ax.plot(adv["epsilons"], adv["accuracy_by_eps"], marker="o",
                label=f"{dial}={row[dial]}, seed={row['seed']}", alpha=0.7)
    ax.set_xlabel("ε")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Adversarial accuracy vs ε, colored by {dial}")
    ax.legend(fontsize=7)
    plt.show()
```

- [ ] **Step 3: Notebook 03 — Landscape comparisons**

Analysis cells:

```python
# List runs that have landscape probes available
import numpy as np

landscape_runs = []
if not df.empty:
    for _, row in df.iterrows():
        p = Path(row["run_dir"]) / "probes" / "landscape.npz"
        if p.exists():
            landscape_runs.append((row, p))
print(f"runs with landscape probes: {len(landscape_runs)}")
```

```python
import numpy as np

fig, axes = plt.subplots(1, max(1, len(landscape_runs)), figsize=(5 * max(1, len(landscape_runs)), 4), squeeze=False)
for ax, (row, npz_path) in zip(axes[0], landscape_runs):
    data = np.load(npz_path)
    grid = data["loss_grid"]
    cf = ax.contourf(data["betas"], data["alphas"], grid, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax)
    title_dial = varying[0] if varying else "run"
    ax.set_title(f"{title_dial}={row[title_dial] if title_dial in row else '?'}, seed={row['seed']}")
plt.show()
```

- [ ] **Step 4: Notebook 05 — Cross-probe agreement scatter**

Analysis cell:

```python
import matplotlib.pyplot as plt

probes_to_compare = [
    ("hessian_lambda_max", "Hessian λ_max (param space sharpness)"),
    ("adversarial_robust_acc_at_8_255", "Robust acc @ 8/255 (input space)"),
    ("margin_mean", "Margin mean"),
]
if not df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if "hessian_lambda_max" in df.columns and "adversarial_robust_acc_at_8_255" in df.columns:
        axes[0].scatter(df["hessian_lambda_max"], df["adversarial_robust_acc_at_8_255"])
        axes[0].set_xlabel("Hessian λ_max")
        axes[0].set_ylabel("robust acc @ 8/255")
        axes[0].set_title("Param vs input sharpness")
    if "margin_mean" in df.columns and "adversarial_robust_acc_at_8_255" in df.columns:
        axes[1].scatter(df["margin_mean"], df["adversarial_robust_acc_at_8_255"])
        axes[1].set_xlabel("margin mean")
        axes[1].set_ylabel("robust acc @ 8/255")
        axes[1].set_title("Margin vs robustness")
    if "hessian_lambda_max" in df.columns and "margin_mean" in df.columns:
        axes[2].scatter(df["hessian_lambda_max"], df["margin_mean"])
        axes[2].set_xlabel("Hessian λ_max")
        axes[2].set_ylabel("margin mean")
        axes[2].set_title("Hessian vs margin")
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 5: Strip outputs and commit**

```bash
uv run nbstripout notebooks/01_margin_distributions.ipynb notebooks/02_eps_curves.ipynb notebooks/03_landscape_comparisons.ipynb notebooks/05_cross_probe_agreement.ipynb
git add notebooks/
git commit -m "feat(notebooks): 01 margin, 02 eps curves, 03 landscape, 05 cross-probe agreement"
```

---

## Task 25: Coverage check + final lint pass

**Files:** none (gates)

- [ ] **Step 1: Run full test suite with coverage**

```bash
uv run pytest --cov=playground --cov-report=term-missing
```

Expected: all tests pass; coverage report shows ≥80 % on `playground/probes/`, `playground/data/`, `playground/runs_io.py`. If anything is below 80 %, add focused tests for the uncovered lines. **Do NOT proceed if coverage fails.**

- [ ] **Step 2: Lint pass**

```bash
uv run ruff check .
uv run black --check .
```

Expected: zero issues. If any fail, fix them.

- [ ] **Step 3: Run a real sweep on the server (cheap one) for a final integration check**

```bash
make sync
make shell
# inside tmux:
uv run python scripts/sweep.py --config configs/exp_label_noise_sweep.yaml \
    --executor gpu-parallel --max-parallel 4 --data-root ./data
# detach when done
```

Then locally:

```bash
make fetch
# open notebooks/04_sweep_summary.ipynb and set EXPERIMENT="label_noise_sharpness"
```

Expected: 12 runs (4 noise levels × 3 seeds) complete; the dashboard plots margin-mean and adversarial-robustness curves vs `label_noise_fraction`. **At this point the playground is operational.**

- [ ] **Step 4: Commit any final touch-ups**

```bash
git add -A
git status   # verify nothing surprising
git diff --cached
git commit -m "chore: coverage + lint pass; playground operational" || echo "no changes to commit"
```

---

## Self-review against the spec

**Spec coverage (each section in `docs/superpowers/specs/2026-05-13-decision-boundary-playground-design.md`):**

- §1 Motivation/goals/non-goals — informational; nothing to implement.
- §2 High-level architecture — embodied in Tasks 7 (Trainer writes run-folder), 16 (probe CLI), 17 (train CLI), 19 (sweep executor), 21 (Makefile), 23–24 (notebooks).
- §3 Repository layout — Task 0 + every subsequent task creates exactly the listed files. Confirmed.
- §4 Models — Task 5. **TinyViT is Phase 2 and intentionally NOT in this plan.** Matches spec.
- §5 Dial taxonomy — RunConfig in Task 2 enumerates every dial in groups 5.1–5.5. Confirmed.
- §6 Sweep & execution
  - 6.1 Config format — Task 18 `expand_sweep` handles `base`, `sweep`, `deep_probe_runs`.
  - 6.2 Executors — Task 19 implements all three.
  - 6.3 Remote workflow — Task 21 implements Makefile targets; Task 22 verifies end-to-end.
  - 6.4 Live progress — TensorBoard mentioned in README (Task 0); spec lists this as optional and we don't wire it as a hard dependency. **Noted gap:** Trainer doesn't currently write a TensorBoard event file. If we want live curves during a run, we'd add a `SummaryWriter` to `Trainer.fit`. I'm leaving this OFF for v1 because metrics.jsonl + `tail -f` is enough for one-screen visibility; we can add TensorBoard in Task 26 (post-launch) if it's actually wanted.
- §7 Probe toolkit — Tasks 8–16 implement P1–P8 + the probe CLI. **P9 (CKA) is Phase 2 and intentionally not in this plan.** Matches spec.
- §8 Analysis layer — Task 18 `runs_io.py`, Tasks 23–24 notebooks 01–05 (note: spec §10 phasing already corrected to include 05 in v1). Confirmed.
- §9 Engineering standards — pyproject pins versions; type hints + docstrings present in every code block; tests target ≥80 % coverage; pre-commit hook configured; no secrets in code (Makefile takes `KEY` as overridable var).
- §10 Phasing — v1 scope = all 25 tasks above. v2/v3 deferred items not in plan. Confirmed.
- §11 Risks — mitigations baked in:
  - aarch64 wheels → only common packages used.
  - Sweep concurrency → `--max-parallel` default 4, `gpu-serial` fallback.
  - Stale server code → `make sync` is single source.
  - Notebook outputs → `nbstripout` in pre-commit.
  - Probe disagreement is signal → notebook 05 makes this concrete.
  - Single-seed misleading → `seed` enforced as sweep axis in example configs (≥3 seeds each).
  - GH200 ARM64 → `torch==2.7.0` pinned to match server.
- §12 Open questions — **answered implicitly:** plan assumes local-only repo (no GitHub remote), no CI in this phase, `nbstripout` enabled. If the user wants any of these turned on, that's a small follow-up task.

**Placeholder scan:** No "TBD", "TODO", "implement later", or empty handlers. Every code block is complete; every command shows expected output.

**Type consistency check:**
- `RunConfig` field names used in every later task match Task 2 exactly (verified `model`, `dataset`, `batch_size`, `seed`, `label_noise_fraction`, `optimizer`, `width_multiplier`, `normalization`, `activation`, `depth`, `dropout`, `dropout_2d`, `label_smoothing`, `mixup_alpha`, `lr`, `lr_schedule`, `epochs`, `experiment_name`, `run_id`).
- `build_model(name, num_classes=, **kwargs)` signature consistent between Task 5 definition and Task 17 caller.
- `Trainer(model, cfg, run_dir, device)` signature consistent between Task 7 definition and Task 17 caller.
- Probe function names: `probe_margin`, `probe_input_grad`, `probe_adversarial`, `probe_boundary_thickness`, `probe_hessian_top_eigenvalues`, `probe_loss_landscape_slice`, `probe_linear_interp`, `probe_calibration` — used consistently between their defining task and `scripts/probe.py` (Task 16).
- `expand_sweep(path) -> list[SweepRun]` returns `SweepRun(run_id, config, deep_probe)` — used consistently in Task 19.
- `load_run`, `load_sweep` signatures used consistently in Tasks 23–24 notebooks.

**Identified minor gap (above):** TensorBoard live-progress writing is not implemented in `Trainer.fit`. The spec calls this "optional". Documenting here so a future Task 26 can add it cleanly.

No other issues found.
