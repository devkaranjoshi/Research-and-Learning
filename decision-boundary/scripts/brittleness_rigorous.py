"""Rigorous brittleness comparison: brittle vs smooth, 3 seeds each, multiple probes,
PGD attack (stronger than FGSM), boundary-thickness probe, and a t-test.

Total: 6 models, 4 probes each. ~25-35 min on a CPU laptop.
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

from playground.config import RunConfig  # noqa: E402
from playground.data import build_dataloaders  # noqa: E402
from playground.models.registry import build_model  # noqa: E402
from playground.probes.adversarial import probe_adversarial  # noqa: E402
from playground.probes.boundary_thickness import probe_boundary_thickness  # noqa: E402
from playground.probes.calibration import probe_calibration  # noqa: E402
from playground.probes.input_grad import probe_input_grad  # noqa: E402
from playground.probes.margin import probe_margin  # noqa: E402
from playground.train.trainer import Trainer  # noqa: E402

OUT = Path("runs/brittleness_rigorous")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"
SEEDS = [0, 1, 2]
EPS_LIST = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]


def _train_and_probe(name: str, cfg: RunConfig) -> dict:
    print(f"  [{name}] training...", flush=True)
    t0 = time.time()
    train_loader, test_loader = build_dataloaders(cfg, Path("./data"))
    model = build_model(cfg.model, dropout=cfg.dropout)
    trainer = Trainer(model, cfg, run_dir=OUT / name, device=DEVICE)
    trainer.fit(train_loader, test_loader)
    print(f"  [{name}] trained in {time.time()-t0:.1f}s; running probes", flush=True)

    margin = probe_margin(model, test_loader, device=DEVICE)
    grad = probe_input_grad(model, test_loader, device=DEVICE)
    # PGD-20: stronger than FGSM, the standard "did you really beat me?" check
    adv = probe_adversarial(
        model, test_loader, device=DEVICE,
        epsilons=EPS_LIST, attack="pgd", pgd_steps=20, pgd_alpha=0.01,
    )
    cal = probe_calibration(model, test_loader, device=DEVICE)
    # Boundary thickness: trim n_directions for speed
    thick = probe_boundary_thickness(
        model, test_loader, device=DEVICE,
        n_directions=4, step=0.05, max_steps=10,
    )
    return {"name": name, "margin": margin, "grad": grad, "adv": adv, "cal": cal, "thick": thick}


def _build_cfg(condition: str, seed: int) -> RunConfig:
    if condition == "brittle":
        return RunConfig(
            model="mlp2", dataset="mnist", optimizer="sgd_momentum",
            lr=0.1, momentum=0.9, weight_decay=0.0,
            batch_size=512, epochs=2, seed=seed, lr_schedule="constant",
            data_augmentation="none", label_smoothing=0.0,
            experiment_name="brittleness_rigorous", run_id=f"brittle_seed{seed}",
        )
    if condition == "smooth":
        return RunConfig(
            model="mlp2", dataset="mnist", optimizer="sgd_momentum",
            lr=0.01, momentum=0.9, weight_decay=5e-4,
            batch_size=32, epochs=2, seed=seed, lr_schedule="constant",
            data_augmentation="none", label_smoothing=0.1,
            experiment_name="brittleness_rigorous", run_id=f"smooth_seed{seed}",
        )
    raise ValueError(condition)


def main() -> None:
    runs: dict[str, list[dict]] = {"brittle": [], "smooth": []}
    overall_t0 = time.time()
    for condition in ["brittle", "smooth"]:
        for seed in SEEDS:
            name = f"{condition}_seed{seed}"
            runs[condition].append(_train_and_probe(name, _build_cfg(condition, seed)))
    print(f"\nTotal wall clock: {(time.time()-overall_t0)/60:.1f} min")

    # ---- Aggregate ----
    summary = {}
    for cond, rs in runs.items():
        clean_acc = np.array([r["adv"]["accuracy_by_eps"][0] for r in rs])
        acc_eps_05 = np.array([r["adv"]["accuracy_by_eps"][2] for r in rs])  # eps=0.05
        acc_eps_2 = np.array([r["adv"]["accuracy_by_eps"][5] for r in rs])   # eps=0.20
        margin_mean = np.array([r["margin"]["mean"] for r in rs])
        margin_p10 = np.array([r["margin"]["p10"] for r in rs])
        thickness = np.array([r["thick"]["mean_thickness"] for r in rs])
        ece = np.array([r["cal"]["ece"] for r in rs])
        grad_norm = np.array([r["grad"]["mean_grad_norm"] for r in rs])
        summary[cond] = {
            "clean_acc": clean_acc, "acc_eps_05": acc_eps_05, "acc_eps_2": acc_eps_2,
            "margin_mean": margin_mean, "margin_p10": margin_p10,
            "thickness": thickness, "ece": ece, "grad_norm": grad_norm,
        }

    # ---- Welch's t-test on the headline metric: robust acc at eps=0.2 ----
    from scipy import stats
    b = summary["brittle"]["acc_eps_2"]
    s = summary["smooth"]["acc_eps_2"]
    t_stat, p_val = stats.ttest_ind(b, s, equal_var=False)
    print("\n=== Statistical test ===")
    print(f"PGD-20 acc @ eps=0.2: brittle={b.mean():.3f}+/-{b.std(ddof=1):.3f},  "
          f"smooth={s.mean():.3f}+/-{s.std(ddof=1):.3f}")
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Decision: {'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'not significant'}")

    # ---- Print full summary table ----
    print("\n=== Per-seed metrics ===")
    print(f"{'metric':<22} {'brittle (3 seeds)':<28} {'smooth (3 seeds)':<28}")
    print("-" * 78)
    for metric_name, key in [
        ("clean acc",        "clean_acc"),
        ("PGD-20 acc@0.05",  "acc_eps_05"),
        ("PGD-20 acc@0.20",  "acc_eps_2"),
        ("margin mean",      "margin_mean"),
        ("margin p10",       "margin_p10"),
        ("boundary thicknes","thickness"),
        ("ECE",              "ece"),
        ("input grad norm",  "grad_norm"),
    ]:
        b_a = summary["brittle"][key]
        s_a = summary["smooth"][key]
        bstr = f"{b_a.mean():.4f} +/- {b_a.std(ddof=1):.4f}"
        sstr = f"{s_a.mean():.4f} +/- {s_a.std(ddof=1):.4f}"
        print(f"{metric_name:<22} {bstr:<28} {sstr:<28}")

    # ---- Plot with error bands ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"brittle": "tab:red", "smooth": "tab:blue"}

    # Panel 1: adversarial curves with shaded std band over seeds
    ax = axes[0, 0]
    for cond, rs in runs.items():
        accs = np.array([r["adv"]["accuracy_by_eps"] for r in rs])  # (3, 7)
        m = accs.mean(0)
        sd = accs.std(0, ddof=1)
        eps = rs[0]["adv"]["epsilons"]
        ax.plot(eps, m, color=colors[cond], marker="o", linewidth=2.5, label=cond)
        ax.fill_between(eps, m - sd, m + sd, color=colors[cond], alpha=0.2)
    ax.set_xlabel("PGD-20 perturbation eps")
    ax.set_ylabel("accuracy")
    ax.set_title("PGD-20 attack robustness (mean +/- 1 std over 3 seeds)\nsteeper drop = more brittle")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: bar chart of headline metrics with error bars
    ax = axes[0, 1]
    metrics = ["clean_acc", "acc_eps_05", "acc_eps_2", "thickness", "ece"]
    metric_labels = ["clean acc", "acc@eps=0.05", "acc@eps=0.20", "thickness", "ECE"]
    x = np.arange(len(metrics))
    width = 0.35
    for i, cond in enumerate(["brittle", "smooth"]):
        means = [summary[cond][m].mean() for m in metrics]
        stds = [summary[cond][m].std(ddof=1) for m in metrics]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
               color=colors[cond], label=cond, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_title("Key metrics (mean +/- 1 std over 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: margin distributions overlaid for all 6 runs
    ax = axes[1, 0]
    for cond, rs in runs.items():
        for j, r in enumerate(rs):
            h = r["margin"]["histogram"]
            edges = np.asarray(h["edges"])
            centers = (edges[:-1] + edges[1:]) / 2
            label = cond if j == 0 else None
            ax.plot(centers, h["counts"], color=colors[cond], alpha=0.6, label=label)
    ax.set_xlabel("logit margin")
    ax.set_ylabel("count")
    ax.set_title("Margin distributions (3 seeds per condition)")
    ax.legend()

    # Panel 4: boundary thickness scatter
    ax = axes[1, 1]
    for cond, rs in runs.items():
        ts = [r["thick"]["mean_thickness"] for r in rs]
        ax.scatter([cond] * len(ts), ts, color=colors[cond], s=80, edgecolor="k", zorder=5)
        ax.scatter([cond], [np.mean(ts)], color=colors[cond], marker="_", s=600, linewidth=4,
                   zorder=4)
    ax.set_ylabel("mean boundary thickness")
    ax.set_title(
        "Boundary thickness per seed\nbar = mean, dots = individual seeds\n(larger = boundary farther in input space)"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Rigorous brittleness: PGD-20, 3 seeds, MLP-2 on MNIST  |  "
        f"Welch's t @eps=0.20: p={p_val:.4f}",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path = OUT / "brittleness_rigorous.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
