# Decision Boundary Playground

> *What makes a neural network's decision boundary sharp and brittle — and what makes it smooth?*

A controlled experimentation rig for studying decision-boundary geometry. Train small
models (MLP, LeNet, small CNN, PreAct-ResNet) on MNIST / CIFAR-10, turn **one dial per
experiment** (batch size, optimizer, SAM ρ, regularization, …), and measure the result with
eight complementary **probes** in both input space and parameter space.

## Probes

| # | Probe | Space |
|---|-------|-------|
| P1 | Margin distribution | input |
| P2 | Input-gradient norm | input |
| P3 | Adversarial ε-curve (FGSM, PGD-k) | input |
| P4 | Boundary thickness (Yang 2020) | input |
| P5 | Hessian top-k eigenvalues (Lanczos) | parameter |
| P6 | Filter-normalized 2D loss landscape (Li 2018) | parameter |
| P7 | Linear interpolation between checkpoints | parameter |
| P8 | Calibration / ECE (Guo 2017) | output |

Details, equations and failure modes: [`docs/PROBES.md`](docs/PROBES.md).

## Experiments & findings

| Exp | Dial | Result |
|-----|------|--------|
| 1 | batch size (fixed LR) | Keskar effect *not* reproduced — confounded by under-fitting at large batch |
| 2 | batch size (Keskar-faithful) | Reproduced: large batch → higher Hessian λ_max and lower PGD robustness |
| 3 | optimizer: SGD / Adam / SAM | SAM flattens the minimum ~1.9×, but robustness gain is marginal on an already-flat baseline |
| 3b | SAM on a sharp baseline | Default ρ = 0.05 under-doses sharp basins |
| 3c | SAM ρ dose-response | Clear optimum (ρ ≈ 0.2, ~+11 pt robustness), with a collapse beyond it — flatness helps only while the minimum still fits the data |

The theory behind these results is in [`docs/THEORY.md`](docs/THEORY.md) and [`docs/SAM_THEORY.md`](docs/SAM_THEORY.md).
Day-by-day logs are in [`docs/runlog/`](docs/runlog/).

## Layout

```
playground/        library: data, models, training, probes, sweep executor, plots
configs/           experiment sweep YAMLs (base.yaml = smooth baseline)
scripts/           train / probe / sweep entry points, analysis & figure generation
tests/             unit + smoke tests
notebooks/         analysis notebooks
docs/              theory, probe reference, dial reference, figures, run logs
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests

# single run
python scripts/train.py --config configs/base.yaml --run-dir runs/adhoc/run_000
python scripts/probe.py --run-dir runs/adhoc/run_000 --data-root ~/data

# a full sweep (train + probes), e.g. the SAM ρ dose-response
python scripts/sweep.py --config configs/exp_optimizer_sam_rho.yaml --data-root ~/data --max-parallel 2
```

Run outputs (checkpoints, metrics, probe JSON) are written to `runs/` and are not tracked
in git. All sweep dials are documented in [`docs/DIALS.md`](docs/DIALS.md).
