# Dial Reference — Decision Boundary Playground

All knobs you can twist in a sweep YAML, grouped by category. Each row shows the **YAML key**, the **type/range**, the **brittle direction** (sharper boundary), the **smooth direction** (flatter boundary), and the **mechanism**.

> Rule: **one dial per experiment.** Hold all others at their default. See `[[research-log-format]]` memory for the per-run log template.

---

## Smooth baseline (the "control" — everything not being swept stays here)

```yaml
model: mlp2                  # or lenet5, small_cnn, preact_resnet20, preact_resnet56
dataset: mnist               # or cifar10
optimizer: sgd_momentum
lr: 0.05
momentum: 0.9
nesterov: false
weight_decay: 5e-4
lr_schedule: cosine
batch_size: 128
sam_rho: 0.05                # only used when optimizer == sam

label_smoothing: 0.1
mixup_alpha: 0.0
cutmix_alpha: 0.0
dropout: 0.0
dropout_2d: 0.0
data_augmentation: none      # use flip_crop on CIFAR by default in practice
stochastic_weight_averaging: false
swa_start_epoch: 0

width_multiplier: 1.0
depth: 20                    # only used by preact_resnet*
normalization: batch_norm
activation: relu

train_subset_fraction: 1.0
label_noise_fraction: 0.0
class_imbalance_factor: 1.0
input_perturbation_sigma: 0.0

adv_training: none
adv_eps: 0.0313725           # 8/255
adv_alpha: 0.0078431          # 2/255

epochs: 20
seed: 0                      # always sweep over [0, 1, 2] for significance
```

---

## Group 1 — Optimization (lands which minimum)

| YAML key | Type | Brittle direction | Smooth direction | Mechanism |
|---|---|---|---|---|
| `optimizer` | `sgd, sgd_momentum, adam, adamw, sam` | `sgd, adam` | **`sam`** | SAM explicitly minimizes worst-case loss in a parameter neighborhood (Foret 2021) |
| `lr` | float (0.001 – 0.5) | very low or very high | moderate (0.05–0.1) | Too-high LR causes loss-surface bouncing; too-low LR stops in a narrow valley |
| `momentum` | float (0.0 – 0.99) | 0 | 0.9 | Momentum smooths the optimization trajectory |
| `nesterov` | bool | false | true | Look-ahead correction reduces overshoot |
| `weight_decay` | float (0.0 – 5e-3) | 0 | **5e-4 to 1e-3** | Penalizes large weights → caps `f`'s slope in input space |
| `lr_schedule` | `constant, cosine, step, warmup_cosine` | `constant` | **`cosine` or `warmup_cosine`** | Annealing into the basin instead of bouncing |
| `batch_size` | int (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) | ≥ 512 | 32 – 128 | Small-batch SGD noise escapes sharp minima (Keskar 2017) |
| `sam_rho` | float (0.01 – 0.2) | small or 0 | 0.05 – 0.1 | Bigger neighborhood → flatter found minimum (but too big → optimization stalls) |
| `gradient_clip` | float \| null | null | 1.0 to 5.0 | Caps gradient explosions that drive sharp curvature |

---

## Group 2 — Regularization (shapes the basin around the minimum)

| YAML key | Type | Brittle direction | Smooth direction | Mechanism |
|---|---|---|---|---|
| `label_smoothing` | float (0.0 – 0.3) | 0 | **0.05 – 0.1** | Caps logit gap → forbids amplification → gentler `f` in input space |
| `mixup_alpha` | float (0.0 – 1.0) | 0 | **0.2 – 1.0** | Trains on convex blends of inputs → linear interpolation of boundary |
| `cutmix_alpha` | float (0.0 – 1.0) | 0 | 0.5 – 1.0 | Patch-swapping variant of mixup |
| `dropout` | float (0.0 – 0.5) | 0 | 0.1 – 0.3 | Removes co-adaptation, less peaky decisions |
| `dropout_2d` | float (0.0 – 0.3) | 0 | 0.1 – 0.2 | Channel-level dropout for conv nets |
| `data_augmentation` | `none, flip_crop, randaugment, autoaugment` | `none` | **`randaugment` or `autoaugment`** (CIFAR), `flip_crop` (MNIST) | Input-space diversity forces smoother boundary |
| `stochastic_weight_averaging` | bool | false | **true** | Averages weights along training trajectory → ensemble smoothing |
| `swa_start_epoch` | int (0 – epochs-1) | n/a | 60% of `epochs` | Start SWA after main training has converged |

---

## Group 3 — Architecture (shapes the loss landscape itself)

| YAML key | Type | Brittle direction | Smooth direction | Mechanism |
|---|---|---|---|---|
| `width_multiplier` | float (0.5 – 4.0) | small (0.5) | 1.5 – 2.0 | Over-parameterization finds flatter minima (deep double descent) |
| `depth` | int (20, 32, 44, 56) — must satisfy `(depth-2) % 6 == 0` for PreActResNet | 20 | 32 – 56 | Deeper = more parameters = flatter, with diminishing returns past ~50 |
| `normalization` | `batch_norm, group_norm, layer_norm, none` | `none` | **`batch_norm`** (or `group_norm` for small batches) | BN's running stats smooth across batches |
| `activation` | `relu, gelu, silu, leaky_relu` | `relu` | **`gelu` or `silu`** | Smooth nonlinearities → smooth gradients → smooth boundary |

---

## Group 4 — Data (changes the objective)

| YAML key | Type | Brittle direction | Smooth direction | Mechanism |
|---|---|---|---|---|
| `train_subset_fraction` | float (0.01 – 1.0) | ≤ 0.1 | 1.0 | Less data → easier memorization → sharper |
| `label_noise_fraction` | float (0.0 – 0.5) | ≥ 0.2 | 0 | Forced memorization carves sharp boundaries through noisy points (Zhang 2017) |
| `class_imbalance_factor` | float (1.0 – 100.0) | ≥ 10 | 1 | Imbalance → overconfident majority-class predictions |
| `input_perturbation_sigma` | float (0.0 – 0.2) | 0 | 0.05 – 0.1 | Training with Gaussian input noise = implicit smoothing |

---

## Group 5 — Adversarial training (direct cure)

| YAML key | Type | Brittle direction | Smooth direction | Mechanism |
|---|---|---|---|---|
| `adv_training` | `none, fgsm, pgd_5, pgd_10` | `none` | **`pgd_5` or `pgd_10`** | Directly optimizes for boundary smoothness in input space |
| `adv_eps` | float | n/a | 8/255 ≈ 0.0314 (CIFAR), 0.3 (MNIST) | Attack budget at train time |
| `adv_alpha` | float | n/a | adv_eps / 4 | Per-step PGD nudge |

⚠ Cost: 2–3× training time, 2–5 pts of clean accuracy.

---

## How to use this when designing a sweep YAML

Pick **one dial** to vary. Use the smooth baseline above for everything else. Run with **at least 3 seeds**.

```yaml
# Example: dose-response curve for label_smoothing
experiment_name: dose_label_smoothing
description: "Does robustness scale with label_smoothing? One-dial sweep."
base:
  model: mlp2
  dataset: mnist
  optimizer: sgd_momentum
  lr: 0.05
  momentum: 0.9
  weight_decay: 5e-4
  lr_schedule: cosine
  batch_size: 128
  data_augmentation: none
  epochs: 20
sweep:
  label_smoothing: [0.0, 0.05, 0.1, 0.2, 0.3]
  seed: [0, 1, 2]
deep_probe_runs:
  - {label_smoothing: 0.0, seed: 0}
  - {label_smoothing: 0.3, seed: 0}
```

Same pattern for any other single dial — just swap which list goes under `sweep:`.

---

## Quick reference — what NOT to combine

These pairs entangle their effects and make causal attribution impossible if you sweep them together. **Vary one, hold the other fixed.**

- `batch_size` × `lr` — large batch + low LR is "effective" SGD; isolate them.
- `optimizer=sam` × `weight_decay` — SAM already smooths; WD double-dips.
- `label_smoothing` × `mixup` — both target overconfidence via different mechanisms.
- `data_augmentation=randaugment` × `train_subset_fraction` — augmentation effectively expands the dataset.
- `adv_training` × `input_perturbation_sigma` — both add input-space noise.

---

## File saved at

`docs/DIALS.md` inside the playground repo. Update this file when new dials are added to `RunConfig`.
