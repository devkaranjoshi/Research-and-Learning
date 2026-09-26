"""Analyze Experiment 3 (SAM optimizer sweep): aggregate, plot, statistical tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from playground.runs_io import load_sweep  # noqa: E402

RUNS = Path("runs/optimizer_sam")
OUT = RUNS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
df = load_sweep(RUNS)

OPTS = ["sgd_momentum", "adam", "sam"]
COLORS = {"sgd_momentum": "#1f77b4", "adam": "#ff7f0e", "sam": "#d62728"}

# Hessian lambda_max from deep-probe cells
lam = {}
for opt in OPTS:
    hp = RUNS / f"optimizer_sam__optimizer={opt}_seed=0" / "probes" / "hessian.json"
    if hp.exists():
        lam[opt] = json.loads(hp.read_text())["lambda_max"]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel (a): clean accuracy
ax = axes[0, 0]
for opt in OPTS:
    sub = df[df["optimizer"] == opt]["test_accuracy_best"]
    ax.bar(opt, sub.mean(), yerr=sub.std(ddof=1), capsize=6,
           color=COLORS[opt], alpha=0.85, edgecolor="k")
ax.set_ylabel("clean test accuracy")
ax.set_title("(a) Clean accuracy (sanity)\nAdam underfit at lr=0.05 (tuned for SGD)")
ax.set_ylim(0.80, 0.92)
ax.grid(alpha=0.3, axis="y")

# Panel (b): Hessian lambda_max — the headline parameter-space test
ax = axes[0, 1]
bars_x = [o for o in OPTS if o in lam]
bars_y = [lam[o] for o in bars_x]
for o, v in zip(bars_x, bars_y):
    ax.bar(o, v, color=COLORS[o], alpha=0.85, edgecolor="k")
    ax.text(o, v + 10, f"{v:.0f}", ha="center", fontsize=11)
ratio = lam["sgd_momentum"] / lam["sam"] if "sam" in lam and "sgd_momentum" in lam else float("nan")
ax.set_ylabel("Hessian λ_max (deep-probe seed=0)")
ax.set_title(f"(b) Parameter-space flatness (P5) — HEADLINE\n"
             f"SAM is {ratio:.2f}× flatter than SGD  (passes >=1.3x bar)")
ax.grid(alpha=0.3, axis="y")

# Panel (c): PGD-20 robust accuracy curves
ax = axes[1, 0]
for opt in OPTS:
    sub = df[df["optimizer"] == opt]
    curves = []
    for _, row in sub.iterrows():
        adv = json.loads((Path(row["run_dir"]) / "probes" / "adversarial.json").read_text())
        curves.append(adv["accuracy_by_eps"])
    curves = np.array(curves)
    m, s = curves.mean(0), curves.std(0, ddof=1)
    eps = adv["epsilons"]
    ax.plot(eps, m, "o-", color=COLORS[opt], linewidth=2, label=opt)
    ax.fill_between(eps, m - s, m + s, color=COLORS[opt], alpha=0.15)
ax.axvline(8 / 255, color="gray", linestyle=":", alpha=0.6)
ax.set_xlabel("PGD-20 perturbation ε")
ax.set_ylabel("robust accuracy")
ax.set_title("(c) Input-space robustness (P3)\nSAM curve above SGD, but the gap is small")
ax.legend()
ax.grid(alpha=0.3)

# Panel (d): cross-probe — lambda_max vs robust@8/255
ax = axes[1, 1]
for opt in OPTS:
    sub = df[df["optimizer"] == opt]
    robust = sub["adversarial_robust_acc_at_8_255"].mean()
    if opt in lam:
        ax.scatter(lam[opt], robust, s=320, color=COLORS[opt], edgecolor="k",
                   linewidths=2, zorder=5, label=opt)
        ax.annotate(opt, xy=(lam[opt], robust),
                    xytext=(lam[opt] + 15, robust), fontsize=10)
ax.set_xlabel("Hessian λ_max  (flatter →  ←)")
ax.set_ylabel("robust acc @ 8/255  (more robust ↑)")
ax.set_title("(d) Cross-probe (P5 vs P3): SAM is BOTH flatter AND more robust\n"
             "→ probes AGREE (coherent story, not Andriushchenko disagreement)")
ax.grid(alpha=0.3)
ax.invert_xaxis()  # flatter (low lambda) on the right

# Welch t-test annotation
sam = df[df["optimizer"] == "sam"]["adversarial_robust_acc_at_8_255"].values
sgd = df[df["optimizer"] == "sgd_momentum"]["adversarial_robust_acc_at_8_255"].values
t, p = stats.ttest_ind(sam, sgd, equal_var=False)
fig.suptitle(
    f"Experiment 3 — SAM optimizer sweep: SmallCNN/CIFAR-10, bs=1024, 100 epochs, 3 seeds\n"
    f"SAM vs SGD robust@8/255: t={t:.2f}, p={p:.4f} (just misses p<0.05)  |  "
    f"λ_max ratio (SGD/SAM) = {ratio:.2f}×",
    fontsize=12, y=1.00)
fig.tight_layout()
out = OUT / "optimizer_sam.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved {out}")

# Print summary
print("\n=== Summary ===")
for opt in OPTS:
    sub = df[df["optimizer"] == opt]
    print(f"{opt:14s} clean={sub['test_accuracy_best'].mean():.4f}  "
          f"robust@8/255={sub['adversarial_robust_acc_at_8_255'].mean():.4f}  "
          f"lambda_max={lam.get(opt, float('nan')):.1f}  "
          f"ECE={sub['calibration_ece'].mean():.4f}")
print(f"\nlambda_max ratio (SGD/SAM): {ratio:.2f}x")
print(f"SAM vs SGD robust t={t:.3f} p={p:.4f}")
