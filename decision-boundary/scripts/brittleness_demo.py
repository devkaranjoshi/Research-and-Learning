"""Train two MNIST classifiers with contrasting recipes; visualize decision brittleness.

Condition A ("brittle"):
    - Large batch (512), high LR, no augmentation, no regularization.
    - Hypothesis: lands in a sharper minimum, brittle decision boundary.

Condition B ("smooth"):
    - Small batch (32), label smoothing 0.1, lower LR.
    - Hypothesis: smoother decision boundary, more adversarially robust.

Both: MLP-2, 3 epochs, seed=0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from playground.config import RunConfig  # noqa: E402
from playground.data import build_dataloaders  # noqa: E402
from playground.models.registry import build_model  # noqa: E402
from playground.probes.adversarial import probe_adversarial  # noqa: E402
from playground.probes.calibration import probe_calibration  # noqa: E402
from playground.probes.input_grad import probe_input_grad  # noqa: E402
from playground.probes.margin import probe_margin  # noqa: E402
from playground.train.trainer import Trainer  # noqa: E402

OUT = Path("runs/brittleness_demo")
OUT.mkdir(parents=True, exist_ok=True)


def run(name: str, cfg: RunConfig, device: str = "cpu") -> dict:
    print(f"\n=== Training {name} ===")
    t0 = time.time()
    train_loader, test_loader = build_dataloaders(cfg, Path("./data"))
    model = build_model(cfg.model, dropout=cfg.dropout)
    trainer = Trainer(model, cfg, run_dir=OUT / name, device=device)
    trainer.fit(train_loader, test_loader)
    print(f"  trained in {time.time()-t0:.1f}s")

    print(f"  computing probes for {name}...")
    margin = probe_margin(model, test_loader, device=device)
    grad = probe_input_grad(model, test_loader, device=device)
    adv = probe_adversarial(
        model,
        test_loader,
        device=device,
        epsilons=[0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3],
        attack="fgsm",
    )
    cal = probe_calibration(model, test_loader, device=device)
    return {"name": name, "margin": margin, "grad": grad, "adv": adv, "cal": cal}


def main() -> None:
    cfg_brittle = RunConfig(
        model="mlp2",
        dataset="mnist",
        optimizer="sgd_momentum",
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=512,
        epochs=3,
        seed=0,
        lr_schedule="constant",
        data_augmentation="none",
        label_smoothing=0.0,
        experiment_name="brittleness_demo",
        run_id="brittle",
    )

    cfg_smooth = RunConfig(
        model="mlp2",
        dataset="mnist",
        optimizer="sgd_momentum",
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        batch_size=32,
        epochs=3,
        seed=0,
        lr_schedule="constant",
        data_augmentation="none",
        label_smoothing=0.1,
        experiment_name="brittleness_demo",
        run_id="smooth",
    )

    results = [run("brittle", cfg_brittle), run("smooth", cfg_smooth)]

    print("\n=== Generating comparison figure ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    colors = {"brittle": "tab:red", "smooth": "tab:blue"}

    ax = axes[0, 0]
    for r in results:
        h = r["margin"]["histogram"]
        edges = np.asarray(h["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, h["counts"], color=colors[r["name"]], label=r["name"], linewidth=2)
        ax.fill_between(centers, h["counts"], alpha=0.2, color=colors[r["name"]])
    ax.set_xlabel("logit margin  f(x)_y - max_{j!=y} f(x)_j")
    ax.set_ylabel("count")
    ax.set_title("Margin distribution\n(narrower = closer to boundary = more brittle)")
    ax.legend()

    ax = axes[0, 1]
    for r in results:
        ax.plot(
            r["adv"]["epsilons"],
            r["adv"]["accuracy_by_eps"],
            marker="o",
            color=colors[r["name"]],
            label=f"{r['name']} (clean acc {r['adv']['accuracy_by_eps'][0]:.3f})",
            linewidth=2,
        )
    ax.set_xlabel("FGSM perturbation eps")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Decision brittleness:\naccuracy drop under adversarial perturbation\n(steeper drop = more brittle)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for r in results:
        h = r["grad"]["histogram"]
        edges = np.asarray(h["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, h["counts"], color=colors[r["name"]], label=r["name"], linewidth=2)
        ax.fill_between(centers, h["counts"], alpha=0.2, color=colors[r["name"]])
    ax.set_xlabel("input-gradient norm  ||grad_x loss||")
    ax.set_ylabel("count")
    ax.set_title("Input-gradient norm\n(higher = steeper loss surface = more brittle)")
    ax.legend()

    ax = axes[1, 1]
    for r in results:
        bins = r["cal"]["bins"]
        stats = r["cal"]["bin_stats"]
        confs = [s["avg_confidence"] for s in stats if s["count"] > 0]
        accs = [s["avg_accuracy"] for s in stats if s["count"] > 0]
        ax.scatter(
            confs,
            accs,
            color=colors[r["name"]],
            s=60,
            edgecolor="k",
            label=f"{r['name']} (ECE={r['cal']['ece']:.4f})",
        )
        ax.plot(confs, accs, color=colors[r["name"]], alpha=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram\n(distance from diagonal = miscalibration)")
    ax.legend()
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)

    fig.suptitle(
        "Decision boundary brittleness: brittle (large-batch SGD, no reg) vs smooth (small batch + label smoothing)\nMLP-2 on MNIST, 3 epochs, seed=0",
        fontsize=13,
        y=1.00,
    )
    fig.tight_layout()
    out_path = OUT / "brittleness_comparison.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']:8s}: clean_acc={r['adv']['accuracy_by_eps'][0]:.3f}")
        print(
            f"          margin_mean={r['margin']['mean']:6.3f}  margin_p10={r['margin']['p10']:6.3f}  "
            f"frac_neg={r['margin']['frac_negative']:.3f}"
        )
        print(f"          grad_norm_mean={r['grad']['mean_grad_norm']:.4f}")
        print(
            f"          acc_at_eps[0.05]={r['adv']['accuracy_by_eps'][2]:.3f}  "
            f"acc_at_eps[0.2]={r['adv']['accuracy_by_eps'][5]:.3f}"
        )
        print(f"          ECE={r['cal']['ece']:.4f}")


if __name__ == "__main__":
    main()
