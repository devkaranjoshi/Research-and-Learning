"""Analyze the batch_size sweep: aggregate, plot, run statistical tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from playground.runs_io import load_sweep  # noqa: E402

RUNS = Path("runs/batch_size_sharpness")
OUT = RUNS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = load_sweep(RUNS)
print(f"Loaded {len(df)} rows.\n")

# ---- Per-batch summary
print("=== Mean ± std by batch_size ===")
print(
    f"{'bs':<7}{'clean_acc':<22}{'robust@8/255':<22}{'margin':<14}{'ECE':<12}"
)
summary = []
for bs in sorted(df["batch_size"].unique()):
    sub = df[df["batch_size"] == bs]
    s = {
        "batch_size": int(bs),
        "clean_acc_mean": sub["test_accuracy_best"].mean(),
        "clean_acc_std": sub["test_accuracy_best"].std(ddof=1),
        "robust_acc_mean": sub["adversarial_robust_acc_at_8_255"].mean(),
        "robust_acc_std": sub["adversarial_robust_acc_at_8_255"].std(ddof=1),
        "margin_mean": sub["margin_mean"].mean(),
        "ece_mean": sub["calibration_ece"].mean(),
    }
    summary.append(s)
    print(
        f"{bs:<7}{s['clean_acc_mean']:.4f}±{s['clean_acc_std']:.4f}     "
        f"{s['robust_acc_mean']:.4f}±{s['robust_acc_std']:.4f}     "
        f"{s['margin_mean']:<14.3f}{s['ece_mean']:<12.4f}"
    )

# ---- Welch's t-test: bs=32 vs bs=8192 on robust_acc
print("\n=== Welch's t-test: bs=32 vs bs=8192 on robust@8/255 ===")
small = df[df["batch_size"] == 32]["adversarial_robust_acc_at_8_255"].values
large = df[df["batch_size"] == 8192]["adversarial_robust_acc_at_8_255"].values
t, p = stats.ttest_ind(small, large, equal_var=False)
print(f"bs=32: {small.mean():.4f}±{small.std(ddof=1):.4f}  vs  bs=8192: {large.mean():.4f}±{large.std(ddof=1):.4f}")
print(f"t = {t:.3f}, p = {p:.4f}  ({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

# ---- Spearman rank correlation
print("\n=== Spearman rank correlation: batch_size vs robust@8/255 ===")
rho, p_rho = stats.spearmanr(df["batch_size"], df["adversarial_robust_acc_at_8_255"])
print(f"rho = {rho:.3f}, p = {p_rho:.4f}")
rho_clean, p_rho_clean = stats.spearmanr(df["batch_size"], df["test_accuracy_best"])
print(f"\n=== Spearman rank correlation: batch_size vs clean_acc ===")
print(f"rho = {rho_clean:.3f}, p = {p_rho_clean:.4f}")

# ---- Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

batch_sizes = sorted(df["batch_size"].unique())
colors_bs = plt.cm.viridis(np.linspace(0, 0.85, len(batch_sizes)))
bs2color = {bs: colors_bs[i] for i, bs in enumerate(batch_sizes)}

# Panel 1: clean accuracy vs batch size
ax = axes[0, 0]
m = [df[df["batch_size"] == bs]["test_accuracy_best"].mean() for bs in batch_sizes]
s = [df[df["batch_size"] == bs]["test_accuracy_best"].std(ddof=1) for bs in batch_sizes]
ax.errorbar(batch_sizes, m, yerr=s, marker="o", linewidth=2, capsize=4, color="tab:blue")
ax.set_xscale("log")
ax.set_xlabel("batch_size")
ax.set_ylabel("clean test accuracy")
ax.set_title("Clean accuracy vs batch_size")
ax.grid(alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)

# Panel 2: PGD robust accuracy at eps=8/255 vs batch size
ax = axes[0, 1]
m = [df[df["batch_size"] == bs]["adversarial_robust_acc_at_8_255"].mean() for bs in batch_sizes]
s = [df[df["batch_size"] == bs]["adversarial_robust_acc_at_8_255"].std(ddof=1) for bs in batch_sizes]
ax.errorbar(batch_sizes, m, yerr=s, marker="o", linewidth=2, capsize=4, color="tab:red")
ax.set_xscale("log")
ax.set_xlabel("batch_size")
ax.set_ylabel("PGD-20 acc @ eps=8/255")
ax.set_title(f"Robust accuracy vs batch_size\nWelch's t (bs=32 vs 8192): p={p:.4f}")
ax.grid(alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)

# Panel 3: full PGD-eps curves overlaid, colored by batch
ax = axes[0, 2]
for bs in batch_sizes:
    sub = df[df["batch_size"] == bs]
    curves = []
    for _, row in sub.iterrows():
        adv = json.loads((Path(row["run_dir"]) / "probes" / "adversarial.json").read_text())
        curves.append(adv["accuracy_by_eps"])
    curves = np.array(curves)
    mean = curves.mean(0)
    std = curves.std(0, ddof=1)
    eps = adv["epsilons"]
    ax.plot(eps, mean, marker="o", color=bs2color[bs], linewidth=2, label=f"bs={bs}")
    ax.fill_between(eps, mean - std, mean + std, color=bs2color[bs], alpha=0.15)
ax.set_xlabel("PGD-20 eps")
ax.set_ylabel("accuracy")
ax.set_title("Full PGD-20 attack curves\n(shaded = ±1 std over 3 seeds)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Panel 4: margin mean vs batch size
ax = axes[1, 0]
m = [df[df["batch_size"] == bs]["margin_mean"].mean() for bs in batch_sizes]
s = [df[df["batch_size"] == bs]["margin_mean"].std(ddof=1) for bs in batch_sizes]
ax.errorbar(batch_sizes, m, yerr=s, marker="o", linewidth=2, capsize=4, color="tab:purple")
ax.set_xscale("log")
ax.set_xlabel("batch_size")
ax.set_ylabel("logit margin mean")
ax.set_title("Logit margin (P1) vs batch_size")
ax.grid(alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)

# Panel 5: ECE
ax = axes[1, 1]
m = [df[df["batch_size"] == bs]["calibration_ece"].mean() for bs in batch_sizes]
s = [df[df["batch_size"] == bs]["calibration_ece"].std(ddof=1) for bs in batch_sizes]
ax.errorbar(batch_sizes, m, yerr=s, marker="o", linewidth=2, capsize=4, color="tab:orange")
ax.set_xscale("log")
ax.set_xlabel("batch_size")
ax.set_ylabel("Expected Calibration Error")
ax.set_title("ECE vs batch_size")
ax.grid(alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)

# Panel 6: drop ratio (clean - robust) / clean — "how much of accuracy is fragile"
ax = axes[1, 2]
drop_ratio = []
drop_std = []
for bs in batch_sizes:
    sub = df[df["batch_size"] == bs]
    drops = (sub["test_accuracy_best"] - sub["adversarial_robust_acc_at_8_255"]) / sub["test_accuracy_best"]
    drop_ratio.append(drops.mean())
    drop_std.append(drops.std(ddof=1))
ax.errorbar(batch_sizes, drop_ratio, yerr=drop_std, marker="o", linewidth=2, capsize=4, color="tab:green")
ax.set_xscale("log")
ax.set_xlabel("batch_size")
ax.set_ylabel("(clean - robust) / clean")
ax.set_title("Fragility ratio\n(fraction of clean accuracy lost to PGD-20)\n*higher = more brittle*")
ax.grid(alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)
ax.set_ylim(0, 1.05)

fig.suptitle(
    f"Batch-size sweep: PreActResNet-20 on CIFAR-10, 20 epochs, 3 seeds each (15 runs total)\n"
    f"Note: LR held at 0.1 across all batch sizes (NOT scaled linearly per Goyal 2017)",
    fontsize=12,
    y=1.00,
)
fig.tight_layout()
out_path = OUT / "batch_size_sweep.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved {out_path}")
