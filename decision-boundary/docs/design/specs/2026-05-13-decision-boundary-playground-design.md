# Decision Boundary Playground — Design Spec

**Status:** Draft for review
**Date:** 2026-05-13
**Owner:** Dev Karan Joshi
**Repository:** `decision-boundary-playground`
**Target hardware:** NVIDIA GH200 480GB (97 GB HBM, compute 9.0, aarch64) on a cloud instance

---

## 1. Motivation

We want a controllable sandbox in which we can **deliberately induce, measure, and mitigate brittle / sharp decision boundaries** in small image classifiers (MNIST, CIFAR-10), so we can build research intuition that later transfers to our pharmacovigilance NER models.

"Sharp" or "brittle" boundaries are regions where a tiny input perturbation flips the model's prediction. They show up in three connected literatures: adversarial robustness (Goodfellow, Madry), loss-landscape geometry (Keskar's "sharp vs. flat minima"), and margin theory. The playground lets us turn dials on the training recipe and dataset, then *quantify* how each dial changes the geometry.

### Goals

1. Twist a single training dial (optimizer, batch size, label noise, etc.) and run a reproducible sweep on the GH200.
2. Measure boundary sharpness through **multiple complementary probes** so we can observe cases where they agree and cases where they disagree.
3. Pull results back to a local laptop and compose comparison figures in notebooks — without re-running anything.
4. Iterate fast: a small CIFAR-10 ResNet should train end-to-end in single-digit minutes on the GH200.

### Non-goals

- Production-grade model serving.
- Reproducing all of adversarial-robustness literature — the playground is a teaching/research tool, not a benchmark suite.
- Text / NER models in this phase (intentionally deferred; see §10).
- Docker on the GPU host (intentionally rejected; see §6.3).

---

## 2. High-level architecture

The system is a **hybrid CLI-trainer + analysis-notebook** ("Pattern C"):

- **GPU server (cloud GH200 instance)** — runs `scripts/train.py`, `scripts/probe.py`, `scripts/sweep.py` headlessly inside a tmux session. Writes everything to `runs/<experiment_name>/<run_id>/`.
- **Local laptop (Windows)** — edits code, pushes to server via rsync, fetches `runs/` back, runs Jupyter notebooks under `notebooks/` that read those artifacts and produce comparison plots.

Each run is **self-contained on disk**: resolved config, training-log JSONL, model checkpoints, and all probe outputs live in one folder. Reproducibility is filesystem-as-database — no SQLite, no MLflow server, no wandb required (TensorBoard is optional for live curves).

---

## 3. Repository layout

```
decision-boundary-playground/
├── pyproject.toml                  # uv-managed; pinned deps
├── README.md
├── Makefile                        # sync / fetch / shell targets for the GH200
├── .gitignore                      # runs/, .venv/, notebooks output cells, etc.
├── playground/                     # importable library
│   ├── __init__.py
│   ├── data/                       # dataset loaders + label-noise / subset wrappers
│   ├── models/                     # MLP-2, LeNet-5, SmallCNN, PreActResNet-20/56, TinyViT
│   ├── train/                      # trainer, optimizers (incl. SAM), schedulers, SWA
│   ├── probes/                     # P1..P9 implementations
│   ├── sweep/                      # config loader + sweep runner
│   ├── viz/                        # matplotlib helpers
│   └── runs_io.py                  # load_run, load_sweep (used by notebooks)
├── configs/
│   ├── base.yaml
│   ├── exp_batch_size_sweep.yaml
│   ├── exp_optimizer_sweep.yaml
│   ├── exp_label_noise_sweep.yaml
│   └── exp_sam_vs_baseline.yaml
├── scripts/
│   ├── train.py                    # single-run trainer
│   ├── probe.py                    # run probes on a checkpoint
│   ├── sweep.py                    # sweep orchestrator
│   └── server_setup.sh             # one-shot venv + deps install on the cloud instance
├── notebooks/                      # local-only analysis (Jupyter)
│   ├── 01_margin_distributions.ipynb
│   ├── 02_eps_curves.ipynb
│   ├── 03_landscape_comparisons.ipynb
│   ├── 04_sweep_summary.ipynb
│   └── 05_cross_probe_agreement.ipynb
├── runs/                           # gitignored — produced artifacts
└── tests/                          # pytest, ≥80% coverage on probes + data wrappers
```

**Discipline:** `playground/` is a library. `scripts/` is the only invocable surface on the server. `notebooks/` `import playground` and read `runs/`; they never train.

---

## 4. Models

All built from scratch (no `torchvision` pretrained) so width, depth, normalization, and activation are first-class dials.

| Model | Params | Dataset | Purpose |
|---|---|---|---|
| `MLP-2` | ~50K | MNIST | Sanity baseline; over/underparam regime study |
| `LeNet-5` | ~62K | MNIST | Classic; very fast |
| `SmallCNN` | ~150K | CIFAR-10 | 3 conv blocks; workhorse for fast sweeps |
| `PreActResNet-20` | ~270K | CIFAR-10 | Canonical sharpness-research model (Keskar, Li et al.) |
| `PreActResNet-56` | ~860K | CIFAR-10 | Deeper variant; depth-vs-sharpness experiments |
| `TinyViT` *(Phase 2)* | ~500K | CIFAR-10 | Modern comparison; attention dynamics differ. Not required in v1. |

### Datasets

- **MNIST** — built-in via `torchvision.datasets.MNIST`, normalized to mean/std (0.1307 / 0.3081).
- **CIFAR-10** — built-in via `torchvision.datasets.CIFAR10`, normalized to channel-wise mean/std.
- Dataset wrappers in `playground/data/` expose **`train_subset_fraction`**, **`label_noise_fraction`**, and **`class_imbalance_factor`** as constructor args so they are sweepable dials.

---

## 5. Dial taxonomy

Sweeps vary one **group** at a time. Mixing groups in a single sweep makes the causal arrow ambiguous and is treated as a research foot-gun.

### 5.1 Optimization (which minimum the run lands in)

- `optimizer` ∈ `{sgd, sgd_momentum, adam, adamw, sam}`
- `lr`, `weight_decay`, `momentum`, `nesterov`
- `lr_schedule` ∈ `{constant, cosine, step, warmup_cosine}`
- `batch_size`
- `sam_rho` (only when `optimizer == sam`)
- `gradient_clip` (None or float)

### 5.2 Regularization (how the basin around the minimum is shaped)

- `label_smoothing` ∈ `[0.0, 0.2]`
- `mixup_alpha`, `cutmix_alpha`
- `dropout`, `dropout_2d`
- `data_augmentation` ∈ `{none, flip_crop, randaugment, autoaugment}`
- `stochastic_weight_averaging` (bool) + `swa_start_epoch`
- `weight_decay` (also listed in 5.1 — relevant in both views)

### 5.3 Architecture (the loss landscape itself)

- `width_multiplier`
- `depth` (block count for ResNets)
- `normalization` ∈ `{batch_norm, layer_norm, group_norm, none}`
- `activation` ∈ `{relu, gelu, silu, leaky_relu}`

### 5.4 Data (the objective being optimized)

- `train_subset_fraction` ∈ `(0, 1]`
- `label_noise_fraction` ∈ `[0, 0.5]`
- `class_imbalance_factor` ≥ 1
- `input_perturbation_sigma` (gaussian noise injection at train time)

### 5.5 Adversarial training (optional)

- `adv_training` ∈ `{none, fgsm, pgd_5, pgd_10}`
- `adv_eps`, `adv_alpha`

---

## 6. Sweep & execution

### 6.1 Config format

```yaml
# configs/exp_batch_size_sweep.yaml
experiment_name: batch_size_sharpness
description: "Keskar 2017 reproduction — does large batch ⇒ sharp minima?"
base:
  model: preact_resnet20
  dataset: cifar10
  optimizer: sgd_momentum
  lr: 0.1
  weight_decay: 5e-4
  epochs: 100
  seed: 0
sweep:
  batch_size: [32, 128, 512, 2048, 8192]
  seed: [0, 1, 2]            # 3 seeds × 5 batch sizes = 15 runs
deep_probe_runs:             # cells that additionally get P5–P7 (expensive)
  - {batch_size: 32, seed: 0}
  - {batch_size: 8192, seed: 0}
```

Rules:
- **`seed` is always a sweep axis** with at least 3 values. No single-seed reporting.
- Sweep grid = Cartesian product of `sweep:` entries.
- `deep_probe_runs` is an explicit allowlist for expensive probes (P5, P6, P7); other runs get only the cheap probes.

### 6.2 Executors

`scripts/sweep.py --config configs/<name>.yaml --executor <mode> --max-parallel N`

| Executor | Use case |
|---|---|
| `local-serial` | Smoke tests / debugging on the laptop (CPU only). |
| `gpu-parallel` | **Default on GH200.** Launches up to `--max-parallel` runs concurrently. Default `N=4`. Each process pins `torch.set_num_threads(4)` to avoid CPU thrash on 64 cores. |
| `gpu-serial` | Forces one run at a time. Use for ResNet-56 or deep-probe runs. |

Concurrency strategy: plain process-level parallelism with all processes sharing the GPU. The GH200's 97 GB HBM easily fits 4–8 SmallCNN/ResNet-20 runs at the batch sizes we sweep. No CUDA MPS needed; revisit if we ever oversubscribe.

### 6.3 Remote workflow (cloud GH200 instance)

```
1. make sync     # rsync code → server (excludes runs/, .venv)
2. make shell    # ssh + attach/create tmux session named "playground"
3. (inside tmux) python scripts/sweep.py --config configs/<name>.yaml
4. make fetch    # rsync runs/ ← server
5. open notebooks/04_sweep_summary.ipynb locally
```

Makefile snippet:

```makefile
SERVER := ubuntu@<server-ip>
KEY    := ~/.ssh/<key>.pem
REMOTE := /home/ubuntu/decision-boundary-playground

sync:
	rsync -avz --exclude runs/ --exclude .venv -e "ssh -i $(KEY)" ./ $(SERVER):$(REMOTE)/

fetch:
	rsync -avz -e "ssh -i $(KEY)" $(SERVER):$(REMOTE)/runs/ ./runs/

shell:
	ssh -i $(KEY) -t $(SERVER) "tmux new -A -s playground -c $(REMOTE)"
```

**No Docker on the GPU host.** A research codebase that mutates frequently is a poor fit for container rebuilds; `uv venv` on the server gives instant iteration. Docker may be added later only to publish a frozen artifact.

### 6.4 Live progress

TensorBoard on the server, viewed locally via SSH port-forward:

```
ssh -i ~/.ssh/<key>.pem -L 6006:localhost:6006 ubuntu@<server-ip>
# then open http://localhost:6006 on the laptop
```

Trackio / wandb can be added later if needed.

---

## 7. Probe toolkit

Each probe is a small pure function in `playground/probes/` returning a `dict` and (optionally) a PNG. Outputs land in `runs/<exp>/<run_id>/probes/`.

| # | Probe | Measures | Cost | Output |
|---|---|---|---|---|
| **P1** | Margin distribution | `f(x)_y − max_{j≠y} f(x)_j` per sample on test set | Cheap (1 fwd pass) | `margin.json` (histogram + percentiles), `margin.png` |
| **P2** | Input-gradient norm | `‖∇_x L(x, y)‖_2` averaged over test set | Cheap | `grad.json` (scalar + per-sample histogram) |
| **P3** | Adversarial ε-curve | Accuracy under FGSM and PGD-10 at ε ∈ {0, 1, 2, 4, 8, 16}/255 | Medium | `adversarial.json` + `adversarial.png` (curve) |
| **P4** | Boundary thickness (Yang et al. 2020) | Mean distance from x before prediction flips, over random directions | Medium | `thickness.json` |
| **P5** | Hessian top-k eigenvalues (parameter space) | Top-5 eigenvalues + trace via Lanczos / Hessian-vector products | **Expensive** | `hessian.json` |
| **P6** | Filter-normalized loss landscape slice (Li et al. 2018) | 51×51 grid of loss along two filter-normalized parameter directions | **Expensive** (~2.6K fwd passes) | `landscape.npz` + `landscape.png` |
| **P7** | Linear-interpolation curve | Loss along the line between two checkpoints | Cheap–Medium | `interp.json` + `interp.png` |
| **P8** | Calibration (ECE + reliability) | Expected Calibration Error | Cheap | `calibration.json` + `calibration.png` |
| **P9** | CKA representation similarity | CKA between two runs' layer activations. **Phase 2.** | Medium | `cka.json` + `cka.png` |

Defaults:
- Every run gets P1, P2, P3, P8.
- Runs listed under `deep_probe_runs:` additionally get P4, P5, P6, P7.
- P9 is invoked manually from a notebook on a pair of runs.

Implementation notes:
- **No external loss-landscape library** — we implement P5/P6 ourselves so we don't depend on aarch64-incompatible wheels. P5 uses `torch.autograd.grad`-based Hessian-vector products + Lanczos (`scipy.sparse.linalg.eigsh` operating on a `LinearOperator`).
- All probes accept a `device` kwarg and run on GPU when available.

---

## 8. Analysis layer (local notebooks)

`playground/runs_io.py` exposes:

```python
def load_run(run_dir: Path) -> Run:
    """Return a dataclass with .config, .metrics_df, .probes (dict of dicts)."""

def load_sweep(experiment_dir: Path) -> pd.DataFrame:
    """One row per run; columns = all config dials + summary metrics + probe scalars."""
```

`load_sweep` is the workhorse — every notebook calls it once, then operates on the dataframe.

| Notebook | What it answers |
|---|---|
| `01_margin_distributions.ipynb` | Overlay margin PDFs across a sweep, color by dial value |
| `02_eps_curves.ipynb` | Accuracy-vs-ε curves; robust-accuracy@8/255 per dial value |
| `03_landscape_comparisons.ipynb` | Side-by-side P6 contour plots for selected deep-probe runs |
| `04_sweep_summary.ipynb` | One-stop dashboard for a single sweep YAML; auto-renders key plots |
| `05_cross_probe_agreement.ipynb` | Scatter Hessian-λ_max (P5) vs. adversarial robust-acc (P3) vs. margin median (P1) — the research-interesting "do these probes agree?" view |

Output discipline:
- Notebooks are committed with **outputs stripped** via `nbstripout` (pre-commit hook). The `.ipynb` source is the artifact under version control; rendered plots saved to `figures/` go in commits only when intentional.
- All notebooks must run end-to-end against an empty `runs/` (showing a friendly "no runs found" message) so they can be committed without artifacts.

---

## 9. Engineering standards

Engineering standards followed:

- Python 3.10+ (server has 3.10.12; pin to ≥3.10, <3.13).
- Type hints on every public function.
- Google-style docstrings.
- Formatting: `black` + `ruff`. Pre-commit hook runs both.
- Tests: `pytest`, **≥80 % coverage** on `playground/probes/`, `playground/data/`, and `playground/runs_io.py`. Models and trainer have smoke tests (one tiny run, asserting loss decreases).
- TDD: write tests first for probes (the most critical, math-heavy code).
- Logging via `logging` module (no `print` in `playground/`).
- No secrets in source. SSH key path lives in `Makefile` as a default variable but is overridable via env var.
- Immutability: training config is a frozen dataclass; mutating it raises.

---

## 10. Phasing

**v1 (this spec):** MLP-2, LeNet-5, SmallCNN, PreActResNet-20/56. Probes P1–P8. Executors `local-serial`, `gpu-parallel`, `gpu-serial`. Notebooks 01–05 (notebook 05 uses only v1 probes). Three example sweeps: batch size, optimizer, label noise.

**v2 (deferred):** TinyViT, probe P9 (CKA representation similarity across runs), adversarial training as a sweepable dial, optional Trackio integration, optional Dockerfile for shareable reproduction.

**v3 (later, conditional):** transfer the playground patterns to a text-classification or NER toy. Boundary visualization is harder in text; this is its own research effort.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| aarch64 wheel availability for some scientific Python libs | All probes implemented in-house; depend only on PyTorch, numpy, scipy, matplotlib, pyyaml — all of which have aarch64 wheels. |
| Sweep concurrency causes GPU OOM or contention | Cap `--max-parallel` (default 4); `gpu-serial` fallback; document VRAM budget per model in README. |
| Stale code on server vs. laptop | `make sync` is the single source of truth; the README forbids editing on the server. |
| Notebooks committed with huge outputs | `nbstripout` pre-commit hook strips outputs before commit. |
| Probes disagree → confusion about "which one is right" | This is **expected** and the entire point of notebook 05. Document in README that probe disagreement is signal, not noise. |
| Single-seed results misleading | `seed` is mandatory as a sweep axis with ≥3 values; sweep summary always shows error bars / per-seed dots. |
| GH200 ARM64 PyTorch 2.7 bugs | Pin `torch==2.7.0` (the version already on server); CI smoke test on every PR using a CPU-only docker image to catch portability regressions. |

---

## 12. Open questions for the user

- **Git remote?** This spec assumes the repo lives locally and may or may not be pushed to GitHub. If yes, name + visibility?
- **CI?** Do we want GitHub Actions running `pytest` + `ruff` on PRs, or is local-only fine for now?
- **`nbstripout`?** Confirm we want this pre-commit hook (recommended) — if you'd rather keep notebook outputs in git, we'll skip it.
