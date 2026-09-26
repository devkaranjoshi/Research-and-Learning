"""Plot the Keskar faithful reproduction results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from playground.runs_io import load_sweep  # noqa: E402

RUNS = Path("runs/keskar_faithful")
OUT = RUNS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
df = load_sweep(RUNS)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
colors = {256: "tab:blue", 5000: "tab:red"}

# Panel 1: clean accuracy bar
ax = axes[0, 0]
for bs in [256, 5000]:
    sub = df[df["batch_size"] == bs]["test_accuracy_best"]
    ax.bar(str(bs), sub.mean(), yerr=sub.std(ddof=1), color=colors[bs], capsize=8, alpha=0.85)
ax.set_ylabel("clean test accuracy")
ax.set_title("Generalization gap (Keskar 2017 main claim)")
ax.set_ylim(0.7, 0.9)
ax.grid(alpha=0.3, axis="y")

# Panel 2: PGD-20 epsilon curves
ax = axes[0, 1]
for bs in [256, 5000]:
    sub = df[df["batch_size"] == bs]
    curves = []
    for _, row in sub.iterrows():
        adv = json.loads((Path(row["run_dir"]) / "probes" / "adversarial.json").read_text())
        curves.append(adv["accuracy_by_eps"])
    curves = np.array(curves)
    m = curves.mean(0); s = curves.std(0, ddof=1)
    eps = adv["epsilons"]
    ax.plot(eps, m, marker="o", color=colors[bs], linewidth=2.5, label=f"bs={bs}")
    ax.fill_between(eps, m - s, m + s, color=colors[bs], alpha=0.2)
ax.set_xlabel("PGD-20 eps"); ax.set_ylabel("accuracy")
ax.set_title("PGD-20 attack curves\n(small batch retains 10% robust acc, large drops to ~0)")
ax.legend(); ax.grid(alpha=0.3)

# Panel 3: Hessian top-5 eigenvalues (deep-probe cells)
ax = axes[0, 2]
for bs in [256, 5000]:
    seed0_dir = RUNS / f"keskar_faithful__batch_size={bs}_seed=0"
    h = json.loads((seed0_dir / "probes" / "hessian.json").read_text())
    idxs = np.arange(1, len(h["top_eigenvalues"]) + 1)
    ax.plot(idxs, h["top_eigenvalues"], marker="o", color=colors[bs], linewidth=2.5,
            markersize=10, label=f"bs={bs}, lambda_max={h['lambda_max']:.0f}")
ax.set_xlabel("eigenvalue rank")
ax.set_ylabel("Hessian eigenvalue (log scale)")
ax.set_yscale("log")
ax.set_title("Top-5 Hessian eigenvalues (parameter-space sharpness)\nlarge batch is ~95x sharper at lambda_max")
ax.legend(); ax.grid(alpha=0.3, which="both")

# Panel 4: margin distribution
ax = axes[1, 0]
for bs in [256, 5000]:
    for _, row in df[df["batch_size"] == bs].iterrows():
        margin = json.loads((Path(row["run_dir"]) / "probes" / "margin.json").read_text())
        h = margin["histogram"]
        edges = np.asarray(h["edges"]); centers = (edges[:-1] + edges[1:]) / 2
        label = f"bs={bs}" if row["seed"] == 0 else None
        ax.plot(centers, h["counts"], color=colors[bs], alpha=0.6, label=label)
ax.set_xlabel("logit margin"); ax.set_ylabel("count")
ax.set_title("Margin distributions (3 seeds each)")
ax.legend()

# Panel 5: landscape contour for bs=256
ax = axes[1, 1]
land = np.load(RUNS / "keskar_faithful__batch_size=256_seed=0" / "probes" / "landscape.npz")
cf = ax.contourf(land["betas"], land["alphas"], land["loss_grid"], levels=25, cmap="viridis")
fig.colorbar(cf, ax=ax)
ax.set_xlabel("beta"); ax.set_ylabel("alpha")
ax.set_title("Loss landscape bs=256 (the 'flat' minimum)")

# Panel 6: landscape contour for bs=5000
ax = axes[1, 2]
land = np.load(RUNS / "keskar_faithful__batch_size=5000_seed=0" / "probes" / "landscape.npz")
cf = ax.contourf(land["betas"], land["alphas"], land["loss_grid"], levels=25, cmap="viridis")
fig.colorbar(cf, ax=ax)
ax.set_xlabel("beta"); ax.set_ylabel("alpha")
ax.set_title("Loss landscape bs=5000 (the 'sharp' minimum)")

fig.suptitle("Keskar 2017 faithful reproduction: SmallCNN on CIFAR-10, Adam, 100 epochs, 3 seeds\n"
             "Small batch (256) vs Large batch (5000, 10% of train set)", fontsize=13, y=1.00)
fig.tight_layout()
out = OUT / "keskar_faithful.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved {out}")
