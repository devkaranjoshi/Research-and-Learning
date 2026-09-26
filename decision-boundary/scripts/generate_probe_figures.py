"""Generate illustrative figures for docs/PROBES_THEORY.md.

Uses 2D synthetic data (two-moons) so that decision boundaries, gradients, and
perturbations are directly visualizable. Each probe gets one or more dedicated
figures saved into docs/figures/.

Run: uv run python scripts/generate_probe_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Color scheme used throughout
BRITTLE_COLOR = "#d62728"  # red
SMOOTH_COLOR = "#2ca02c"  # green
NEUTRAL = "#666666"

torch.manual_seed(0)
np.random.seed(0)


# -------------------------------------------------------------------
# Small 2D classifier we can train in seconds
# -------------------------------------------------------------------
class MLP2D(nn.Module):
    def __init__(self, hidden: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_2d_model(x_train, y_train, epochs=400, lr=0.05, weight_decay=0.0,
                   width=64, label_smoothing=0.0, seed=0):
    """Train a small MLP on 2D data. Returns the model."""
    torch.manual_seed(seed)
    model = MLP2D(hidden=width)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def make_data(n_samples: int = 200, noise: float = 0.15):
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    # Normalize to roughly [-1, 1]
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float32), y.astype(np.int64)


def plot_decision_boundary(ax, model, x, y, resolution=300,
                           xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                           show_data=True, alpha=0.5):
    """Render the decision boundary as a heatmap behind the data."""
    xx, yy = np.meshgrid(np.linspace(*xlim, resolution),
                         np.linspace(*ylim, resolution))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(grid))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy().reshape(resolution, resolution)
    cm = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=alpha, vmin=0, vmax=1)
    # The decision boundary itself: probs == 0.5
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
    if show_data:
        ax.scatter(x[y == 0, 0], x[y == 0, 1], c="tab:blue", s=20,
                   edgecolor="k", linewidths=0.5, label="class 0")
        ax.scatter(x[y == 1, 0], x[y == 1, 1], c="tab:red", s=20,
                   edgecolor="k", linewidths=0.5, label="class 1")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    return cm


# ==================================================================
# Figure 0 — what is a decision boundary? (intro figure)
# ==================================================================
def fig_intro_boundary():
    x, y = make_data()
    model = train_2d_model(x, y, epochs=400, weight_decay=1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: data only
    ax = axes[0]
    ax.scatter(x[y == 0, 0], x[y == 0, 1], c="tab:blue", s=30,
               edgecolor="k", linewidths=0.5, label="class 0")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c="tab:red", s=30,
               edgecolor="k", linewidths=0.5, label="class 1")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title("(a) The data — a classifier's input")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    # Panel B: data + boundary
    ax = axes[1]
    plot_decision_boundary(ax, model, x, y)
    ax.set_title("(b) The decision boundary (black) + soft-prob field")
    ax.grid(alpha=0.2)

    fig.suptitle("A 2D classifier on two-moons data. Black curve = where p(class=1|x) = 0.5.",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "00_intro_boundary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 00_intro_boundary.png")


# ==================================================================
# Figure 1 — P1 margin: what does "margin" mean geometrically?
# ==================================================================
def fig_p1_margin():
    x, y = make_data()
    model_brittle = train_2d_model(x, y, epochs=400, weight_decay=0.0,
                                   width=128, label_smoothing=0.0, lr=0.1)
    model_smooth = train_2d_model(x, y, epochs=400, weight_decay=5e-3,
                                  width=32, label_smoothing=0.1, lr=0.02)

    # 2x2 layout: top row boundary plots; bottom-left histogram, bottom-right annotated geometry
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panels (a, b) — boundary heatmaps
    for col, (model, name) in enumerate(
        [(model_brittle, "brittle (no reg, high lr)"),
         (model_smooth, "smooth (wd + label smoothing)")]):
        ax = axes[0, col]
        plot_decision_boundary(ax, model, x, y, alpha=0.5)
        ax.set_title(f"({chr(97+col)}) {name}\nblack = decision boundary",
                     fontsize=11)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # Panel (c) — margin histograms overlaid
    ax = axes[1, 0]
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    for model, name, color in [
        (model_brittle, "brittle", BRITTLE_COLOR),
        (model_smooth, "smooth", SMOOTH_COLOR),
    ]:
        with torch.no_grad():
            logits = model(x_t)
            correct = logits.gather(1, y_t.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y_t.unsqueeze(1), float("-inf"))
            max_other, _ = masked.max(dim=1)
            margins = (correct - max_other).numpy()
        ax.hist(margins, bins=30, alpha=0.55,
                label=f"{name} (mean={margins.mean():.2f})",
                color=color, edgecolor="k")
    ax.set_xlabel("logit margin = f(x)_y − max_{j≠y} f(x)_j", fontsize=10)
    ax.set_ylabel("# samples")
    ax.set_title("(c) Margin distributions on test data\n"
                 "(brittle's wider tail = more overconfidence)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel (d) — geometric interpretation
    ax = axes[1, 1]
    plot_decision_boundary(ax, model_brittle, x, y, alpha=0.35,
                           xlim=(-2.5, 2.5), ylim=(-2.2, 2.2), show_data=False)
    near_idx = np.argmin(np.abs([
        model_brittle(torch.tensor([xi], dtype=torch.float32))[0, 1].item()
        - model_brittle(torch.tensor([xi], dtype=torch.float32))[0, 0].item()
        for xi in x]))
    far_idx = np.argmax([
        abs(model_brittle(torch.tensor([xi], dtype=torch.float32))[0, 1].item()
            - model_brittle(torch.tensor([xi], dtype=torch.float32))[0, 0].item())
        for xi in x])

    for idx, label_text, dxoff in [
        (near_idx, "low margin ≈ 0", (0.8, 0.6)),
        (far_idx, "high margin", (-0.6, -0.7)),
    ]:
        pt = x[idx]
        with torch.no_grad():
            logits = model_brittle(
                torch.tensor([pt], dtype=torch.float32)).numpy()[0]
        m = logits[y[idx]] - logits[1 - y[idx]]
        c = "tab:blue" if y[idx] == 0 else "tab:red"
        ax.scatter(pt[0], pt[1], c=c, s=260, edgecolor="black",
                   linewidths=2.5, zorder=10)
        ax.annotate(f"{label_text}\nm = {m:.1f}",
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0] + dxoff[0], pt[1] + dxoff[1]),
                    fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="black", alpha=0.95))
    ax.set_title("(d) Logit margin is distance in **logit space**, not input space.\n"
                 "Both points sit on the same decision boundary in 2D, but their\n"
                 "logit margins can differ wildly.", fontsize=10.5)
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.grid(alpha=0.2)

    fig.suptitle("Probe P1 — Margin Distribution: confidence in logit space",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "01_p1_margin.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 01_p1_margin.png")


# ==================================================================
# Figure 2 — P2 input-gradient norm
# ==================================================================
def fig_p2_input_grad():
    x, y = make_data()
    model_brittle = train_2d_model(x, y, epochs=400, weight_decay=0.0, width=128, lr=0.1)
    model_smooth = train_2d_model(x, y, epochs=400, weight_decay=5e-3, width=32, lr=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Compute gradient field on a grid
    for col, (model, name, color) in enumerate([
        (model_brittle, "brittle: large |∇_x ℓ|", BRITTLE_COLOR),
        (model_smooth, "smooth: small |∇_x ℓ|", SMOOTH_COLOR),
    ]):
        ax = axes[col]
        # decision boundary
        plot_decision_boundary(ax, model, x, y, alpha=0.4, xlim=(-2.5, 2.5),
                              ylim=(-2.5, 2.5), show_data=True)
        # Quiver plot of gradient direction
        gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 12), np.linspace(-2.2, 2.2, 12))
        grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32),
                            requires_grad=True)
        # Use class 0 as "target" — we want grad of loss for those who are class 0
        target = torch.zeros(grid.shape[0], dtype=torch.long)
        loss = F.cross_entropy(model(grid), target, reduction="sum")
        grad = torch.autograd.grad(loss, grid)[0].numpy()
        # Normalize for display
        norms = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
        u = (grad[:, 0] / norms.ravel()).reshape(gx.shape)
        v = (grad[:, 1] / norms.ravel()).reshape(gy.shape)
        ax.quiver(gx, gy, u, v, color="black", scale=25, width=0.004, alpha=0.7)
        ax.set_title(f"({chr(97+col)}) {name}")
        ax.set_aspect("equal")

    # Panel (c) — histogram of gradient norms on real data
    ax = axes[2]
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    for model, name, color in [(model_brittle, "brittle", BRITTLE_COLOR),
                                (model_smooth, "smooth", SMOOTH_COLOR)]:
        x_req = x_t.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_req), y_t, reduction="sum")
        grad = torch.autograd.grad(loss, x_req)[0]
        norms = grad.norm(p=2, dim=1).detach().numpy()
        ax.hist(norms, bins=30, alpha=0.55, color=color, edgecolor="k",
                label=f"{name} (mean={norms.mean():.3f})")
    ax.set_xlabel("‖∇_x ℓ(f(x), y)‖_2")
    ax.set_ylabel("# samples")
    ax.set_title("(c) Distribution of gradient norms")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Probe P2 — Input-Gradient Norm: how steeply does loss rise around each input?\n"
                 "Arrows show ∇_x ℓ direction; their density is high where loss surface is steep.",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_p2_input_grad.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 02_p2_input_grad.png")


# ==================================================================
# Figure 3 — P3 adversarial attack (FGSM + PGD)
# ==================================================================
def fig_p3_adversarial():
    x, y = make_data()
    model_brittle = train_2d_model(x, y, epochs=400, weight_decay=0.0, width=128, lr=0.1)
    model_smooth = train_2d_model(x, y, epochs=400, weight_decay=5e-3, width=32, lr=0.02)

    # Layout: 2 rows. Top = two ε-ball/attack plots side-by-side.
    # Bottom = single wide accuracy-vs-eps curve.
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95])

    # Pick a sample to attack
    idx_clean = 50
    x_pick = x[idx_clean].copy()
    y_pick = int(y[idx_clean])

    def fgsm_step(model, x_t, y_t, eps):
        x_t = x_t.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_t), y_t)
        grad = torch.autograd.grad(loss, x_t)[0]
        return (x_t + eps * grad.sign()).detach()

    def pgd_steps(model, x_t, y_t, eps, alpha, n_iter=10, x_orig=None):
        if x_orig is None:
            x_orig = x_t.clone()
        x_adv = x_t + torch.empty_like(x_t).uniform_(-eps * 0.3, eps * 0.3)
        path = [x_adv.detach().clone()]
        for _ in range(n_iter):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y_t)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = (x_adv.detach() + alpha * grad.sign())
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            path.append(x_adv.detach().clone())
        return x_adv, torch.stack(path).numpy()

    # Top row: brittle (a) and smooth (b) — ε-ball + attacks visualized
    for col, (model, name) in enumerate([
        (model_brittle, "brittle"),
        (model_smooth, "smooth"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        plot_decision_boundary(ax, model, x, y, alpha=0.4,
                               xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), show_data=True)
        eps = 0.4
        sq = plt.Rectangle((x_pick[0] - eps, x_pick[1] - eps), 2 * eps, 2 * eps,
                            fill=False, edgecolor="black", linewidth=2,
                            linestyle="--")
        ax.add_patch(sq)
        ax.scatter(x_pick[0], x_pick[1], c="yellow", s=220, edgecolor="k",
                   linewidths=2, zorder=10, label="clean x")
        x_t = torch.tensor([x_pick], dtype=torch.float32)
        y_t = torch.tensor([y_pick], dtype=torch.long)
        x_fgsm = fgsm_step(model, x_t, y_t, eps).numpy()[0]
        ax.scatter(x_fgsm[0], x_fgsm[1], c="cyan", s=220, marker="^",
                   edgecolor="k", linewidths=2, zorder=10, label="FGSM ε=0.4")
        _, path = pgd_steps(model, x_t, y_t, eps, alpha=eps / 5, n_iter=10)
        path = path.squeeze(1)
        ax.plot(path[:, 0], path[:, 1], "-", color="magenta",
                linewidth=1.8, alpha=0.8)
        ax.scatter(path[-1, 0], path[-1, 1], c="magenta", s=240, marker="*",
                   edgecolor="k", linewidths=2, zorder=10, label="PGD-10 ε=0.4")
        ax.set_title(f"({chr(97+col)}) {name}\n"
                     f"dashed box = L∞ ε-ball  |  cyan ▲ = FGSM  |  magenta ★ = PGD",
                     fontsize=11)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
        ax.set_aspect("equal")

    # Bottom row: accuracy vs eps curves (full width)
    ax = fig.add_subplot(gs[1, :])
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    for model, name, color in [(model_brittle, "brittle", BRITTLE_COLOR),
                                (model_smooth, "smooth", SMOOTH_COLOR)]:
        accs_fgsm, accs_pgd = [], []
        for eps in epsilons:
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            if eps == 0:
                with torch.no_grad():
                    pred = model(x_t).argmax(1)
                acc = (pred == y_t).float().mean().item()
                accs_fgsm.append(acc); accs_pgd.append(acc)
                continue
            x_a = fgsm_step(model, x_t, y_t, eps)
            with torch.no_grad():
                pred = model(x_a).argmax(1)
            accs_fgsm.append((pred == y_t).float().mean().item())
            x_a, _ = pgd_steps(model, x_t, y_t, eps, alpha=eps / 5, n_iter=10)
            with torch.no_grad():
                pred = model(x_a).argmax(1)
            accs_pgd.append((pred == y_t).float().mean().item())
        ax.plot(epsilons, accs_fgsm, "o--", color=color, alpha=0.6,
                label=f"{name} — FGSM")
        ax.plot(epsilons, accs_pgd, "s-", color=color, linewidth=2.5,
                label=f"{name} — PGD-10")
    ax.set_xlabel("attack budget ε (L_∞)")
    ax.set_ylabel("accuracy under attack")
    ax.set_title("Accuracy vs perturbation budget — the **ε-curve** the probe returns.\n"
                 "Steeper drop = more brittle. PGD always lies below FGSM (stronger attacker).")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    fig.suptitle("Probe P3 — Adversarial ε-Curve (FGSM + PGD): the operational gold standard",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "03_p3_adversarial.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 03_p3_adversarial.png")


# ==================================================================
# Figure 4 — P4 boundary thickness
# ==================================================================
def fig_p4_boundary_thickness():
    x, y = make_data()
    model = train_2d_model(x, y, epochs=400, weight_decay=5e-3, width=64)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel (a) — illustrate the probe mechanism
    ax = axes[0]
    plot_decision_boundary(ax, model, x, y, alpha=0.35, show_data=True)
    # Pick a sample point that's confidently classified
    idx = 5
    pt = x[idx]
    ax.scatter(pt[0], pt[1], c="yellow", s=300, edgecolor="k", linewidths=2,
               zorder=10, label="sample x")
    # Draw n random rays from the sample
    n_rays = 10
    np.random.seed(7)
    step = 0.05
    max_steps = 80
    flip_distances = []
    with torch.no_grad():
        original_pred = model(torch.tensor([pt], dtype=torch.float32)).argmax(1).item()
    for r in range(n_rays):
        theta = 2 * np.pi * r / n_rays
        direction = np.array([np.cos(theta), np.sin(theta)])
        # Walk along this ray and find flip
        for k in range(1, max_steps + 1):
            x_perturbed = pt + k * step * direction
            with torch.no_grad():
                pred = model(torch.tensor([x_perturbed],
                                          dtype=torch.float32)).argmax(1).item()
            if pred != original_pred:
                flip_distances.append(k * step)
                # Plot ray up to flip point
                ax.plot([pt[0], x_perturbed[0]], [pt[1], x_perturbed[1]],
                        "-", color="purple", linewidth=1.5, alpha=0.7)
                ax.scatter(x_perturbed[0], x_perturbed[1], c="purple", s=50,
                           edgecolor="k", linewidths=0.5, zorder=10)
                break
        else:
            flip_distances.append(max_steps * step)
            x_end = pt + max_steps * step * direction
            ax.plot([pt[0], x_end[0]], [pt[1], x_end[1]],
                    ":", color="purple", linewidth=1, alpha=0.4)

    thickness = float(np.mean(flip_distances))
    ax.set_title(f"(a) For each random direction, walk until prediction flips.\n"
                 f"Boundary thickness ≈ mean flip distance = {thickness:.2f}",
                 fontsize=11)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Panel (b) — explain the geometry
    ax = axes[1]
    ax.axis("off")
    ax.text(0.05, 0.95, "What boundary thickness measures",
            fontsize=14, weight="bold")
    ax.text(0.05, 0.80,
            "For each sample x:\n"
            "  1. Pick n random unit directions d_i\n"
            "  2. For each direction, walk step·k along d_i\n"
            "     until argmax f(x + step·k·d_i) ≠ argmax f(x)\n"
            "  3. Record the flip distance t_i*\n\n"
            "Thickness(x) = mean over i of min(t_i*, max_steps·step)\n\n"
            "(If no flip within max_steps, the distance is capped at\n"
            "max_steps·step — a *floor* on robustness)",
            fontsize=11, family="monospace", verticalalignment="top")
    ax.text(0.05, 0.30,
            "Intuition:\n"
            "  Larger thickness = boundary is farther in input space\n"
            "  = more attacker work needed to flip any prediction\n"
            "  = robust\n\n"
            "vs P3 (adversarial):\n"
            "  P3 finds worst-case δ → measures worst direction\n"
            "  P4 averages over random directions → measures typical\n"
            "  Together they cover both extremes.",
            fontsize=10.5, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#fff7e0", edgecolor="black"))

    fig.suptitle("Probe P4 — Boundary Thickness: mean flip-distance over random rays",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "04_p4_boundary_thickness.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 04_p4_boundary_thickness.png")


# ==================================================================
# Figure 5 — P5 Hessian: sharp vs flat minimum
# ==================================================================
def fig_p5_hessian():
    # 1x2 layout: 1D sharp vs flat + Hessian spectrum from real Exp 2 data.
    # The Pearlmutter explanation moves into the markdown doc (text doesn't
    # render well in a matplotlib panel without overlapping).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a) — 1D sharp vs flat minima cartoon
    ax = axes[0]
    theta = np.linspace(-3, 3, 200)
    L_flat = 0.1 * theta**2 + 0.5
    L_sharp = 2.5 * theta**2 + 0.5
    ax.plot(theta, L_flat, color=SMOOTH_COLOR, linewidth=2.5,
            label="flat:  λ_max = 0.2")
    ax.plot(theta, L_sharp, color=BRITTLE_COLOR, linewidth=2.5,
            label="sharp: λ_max = 5.0")
    delta = 0.5
    for L_arr, c, dy, side in [(L_flat, SMOOTH_COLOR, 0.6, "right"),
                                (L_sharp, BRITTLE_COLOR, -0.8, "left")]:
        L_at_delta = L_arr[np.argmin(np.abs(theta - delta))]
        L_min = L_arr.min()
        ax.plot([0, delta], [L_min, L_min], "k:", alpha=0.5)
        ax.plot([delta, delta], [L_min, L_at_delta], "k:", alpha=0.5)
        ax.annotate(f"ΔL = {L_at_delta - L_min:.2f}",
                    xy=(delta, (L_at_delta + L_min) / 2),
                    xytext=(delta + (0.5 if side == "right" else -1.5),
                            (L_at_delta + L_min) / 2 + dy),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white",
                              alpha=0.9, edgecolor="gray"))
    ax.axvline(delta, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("perturbation along top eigenvector (Δθ)")
    ax.set_ylabel("L(θ* + Δθ)")
    ax.set_title("(a) 1D analogue: λ_max controls how fast loss rises\n"
                 "Same perturbation Δθ → very different ΔL", fontsize=11)
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)

    # Panel (b) — Hessian spectrum: sharp vs flat (real data from Exp 2)
    ax = axes[1]
    sharp_eigs = [9773, 6187, 5053, 4651, 3422]
    flat_eigs = [103, 89, 73, 66, 55]
    idx = np.arange(1, 6)
    ax.semilogy(idx, sharp_eigs, "o-", color=BRITTLE_COLOR, linewidth=2.5,
                markersize=12, label="sharp (bs=5000, λ_max = 9 773)")
    ax.semilogy(idx, flat_eigs, "o-", color=SMOOTH_COLOR, linewidth=2.5,
                markersize=12, label="flat  (bs=256,   λ_max =   103)")
    for i, (s, f) in enumerate(zip(sharp_eigs, flat_eigs)):
        ax.text(i + 1, s * 1.5, f"{s:,}", color=BRITTLE_COLOR,
                fontsize=9, ha="center")
        ax.text(i + 1, f * 0.55, f"{f}", color=SMOOTH_COLOR,
                fontsize=9, ha="center")
    ax.set_xlabel("eigenvalue rank (1 = largest)")
    ax.set_ylabel("Hessian eigenvalue (log scale)")
    ax.set_title("(b) Top-5 Hessian eigenvalues from Exp 2 (real measurements)\n"
                 "95× ratio at λ_max; ~60× at each subsequent rank", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(30, 30000)

    fig.suptitle("Probe P5 — Hessian Top-k Eigenvalues (parameter-space sharpness gold standard)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "05_p5_hessian.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 05_p5_hessian.png")


# ==================================================================
# Figure 6 — P6 landscape probe + the span issue
# ==================================================================
def fig_p6_landscape():
    """Synthetic quadratic-bowl + saddle to illustrate the span-tuning issue."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def make_grid(span, size=51):
        a = np.linspace(-span, span, size)
        b = np.linspace(-span, span, size)
        A, B = np.meshgrid(a, b)
        return A, B

    # Two synthetic basins with different curvature
    flat_lambda = 1.0       # gentle basin
    sharp_lambda = 100.0    # steep basin
    L_min = 0.05            # base loss at minimum

    # Demo: show that span=1 looks similar; span tuned to basin width shows difference
    spans = [(1.0, "default span = 1.0 (too wide for sharp basin)"),
             (0.3, "span = 0.3 (basin visible for flat)"),
             (None, "auto-tuned span = c/√λ_max  *  (basin centered in view)")]

    L_ceiling = 2.30  # log(10) ceiling for CIFAR-10 cross-entropy

    for col, (s, title) in enumerate(spans):
        for row, (lam, name, color) in enumerate(
            [(flat_lambda, f"flat basin (λ_max = {flat_lambda:.0f})", SMOOTH_COLOR),
             (sharp_lambda, f"sharp basin (λ_max = {sharp_lambda:.0f})", BRITTLE_COLOR)]):
            ax = axes[row, col]
            if s is None:
                # Auto-tune
                span_used = 3.0 / np.sqrt(lam)
            else:
                span_used = s
            A, B = make_grid(span_used, size=51)
            # 2D quadratic + ceiling saturation
            L = L_min + 0.5 * lam * (A**2 + B**2)
            L = np.minimum(L, L_ceiling)
            cf = ax.contourf(B, A, L, levels=20, cmap="viridis", vmin=L_min, vmax=L_ceiling)
            ax.set_xlim(-span_used, span_used); ax.set_ylim(-span_used, span_used)
            ax.set_aspect("equal")
            ax.set_title(f"{name}\nspan used: ±{span_used:.3f}", fontsize=10)
            ax.set_xlabel("β"); ax.set_ylabel("α")
            if col == 0:
                ax.text(0.02, 0.97, title.split("(")[0], transform=ax.transAxes,
                        fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white"))

    # Add a row title for each column
    for col, (_, title) in enumerate(spans):
        axes[0, col].set_title(title + "\n" + axes[0, col].get_title(), fontsize=10)

    fig.suptitle("Probe P6 — Filter-Normalized Loss Landscape (Li 2018): span tuning matters\n"
                 "Top: flat basin   Bottom: sharp basin (100× larger λ).\n"
                 "Left: default span — both look similar (ceiling dominates). "
                 "Right: auto-tuned span — sharp basin's narrowness becomes visible.",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "06_p6_landscape.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 06_p6_landscape.png")


# ==================================================================
# Figure 7 — P7 linear interpolation between modes
# ==================================================================
def fig_p7_linear_interp():
    # Single big panel showing connected vs disconnected basin signatures
    fig, ax = plt.subplots(figsize=(11, 6))

    alphas = np.linspace(0, 1, 200)
    connected = 0.3 + 0.12 * np.sin(np.pi * alphas)
    bump = 0.3 + 0.35 * np.sin(np.pi * alphas)
    disconnected = 0.3 + 1.5 * np.exp(-((alphas - 0.5) / 0.10) ** 2 * 4)

    ax.plot(alphas, connected, color=SMOOTH_COLOR, linewidth=3,
            label="(I) same basin — smooth, low bump")
    ax.plot(alphas, bump, color="#d6a72c", linewidth=3,
            label="(II) loss-equal modes — moderate bump")
    ax.plot(alphas, disconnected, color=BRITTLE_COLOR, linewidth=3,
            label="(III) different basins — sharp barrier")
    ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5)

    ax.text(0.02, 0.32, "L(θ_A)", fontsize=10, color="gray")
    ax.text(0.95, 0.32, "L(θ_B)", fontsize=10, color="gray", ha="right")

    ax.annotate("barrier height\n= activation cost\nto leave θ_A's basin",
                xy=(0.5, 1.8), xytext=(0.72, 1.6),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="gray", alpha=0.95))

    ax.set_xlabel("interpolation parameter α   "
                  "θ(α) = (1 − α)·θ_A + α·θ_B")
    ax.set_ylabel("loss along the line, L(θ(α))")
    ax.set_title("Probe P7 — Linear Interpolation Between Checkpoints\n"
                 "Three signatures of basin geometry (Garipov 2018 'mode connectivity')",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.25, 2.0)

    fig.tight_layout()
    fig.savefig(OUT / "07_p7_linear_interp.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 07_p7_linear_interp.png")


# ==================================================================
# Figure 8 — P8 calibration / reliability diagram
# ==================================================================
def fig_p8_calibration():
    x, y = make_data(n_samples=400)
    model_overconf = train_2d_model(x, y, epochs=600, weight_decay=0.0,
                                    width=128, label_smoothing=0.0, lr=0.1)
    model_smooth = train_2d_model(x, y, epochs=600, weight_decay=5e-3, width=32,
                                  label_smoothing=0.15, lr=0.02)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Compute reliability diagrams
    def reliability(model, x, y, n_bins=10):
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(model(x_t), dim=1)
            conf, pred = probs.max(dim=1)
        conf = conf.numpy(); pred = pred.numpy(); y_np = y_t.numpy()
        bins = np.linspace(0.5, 1.0, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        bin_conf, bin_acc, bin_n = [], [], []
        ece = 0.0
        N = len(conf)
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                bin_conf.append(None); bin_acc.append(None); bin_n.append(0); continue
            c, a = conf[mask].mean(), (pred[mask] == y_np[mask]).mean()
            bin_conf.append(c); bin_acc.append(a); bin_n.append(mask.sum())
            ece += mask.mean() * abs(c - a)
        return bins, bin_conf, bin_acc, bin_n, ece

    for col, (model, name, color) in enumerate([
        (model_overconf, "overconfident (no smoothing)", BRITTLE_COLOR),
        (model_smooth, "label-smoothed", SMOOTH_COLOR),
    ]):
        ax = axes[col]
        bins, bin_conf, bin_acc, bin_n, ece = reliability(model, x, y, n_bins=10)
        # Bar plot at bin centers
        centers = (bins[:-1] + bins[1:]) / 2
        accs = [a if a is not None else 0 for a in bin_acc]
        confs = [c if c is not None else centers[i] for i, c in enumerate(bin_conf)]
        ax.bar(centers, accs, width=0.045, color=color, alpha=0.8,
                edgecolor="k", label="accuracy / bin")
        # Confidence dots
        for c, a, n in zip(confs, accs, bin_n):
            if n > 0:
                ax.scatter(c, a, c="black", s=60, zorder=10)
        # Diagonal
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="perfect calibration")
        ax.set_xlim(0.5, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence (max softmax)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"({chr(97+col)}) {name}\nECE = {ece:.4f}", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Probe P8 — Calibration / Expected Calibration Error (ECE)\n"
                 "Perfect calibration = bars on diagonal. Above diagonal = under-confident, below = over-confident.",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "08_p8_calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 08_p8_calibration.png")


# ==================================================================
# Figure 9 — Cross-probe agreement / disagreement
# ==================================================================
def fig_cross_probe():
    # Single big scatter plot with quadrant annotations on the figure itself,
    # not in a text panel. Cleaner and avoids overlap.
    fig, ax = plt.subplots(figsize=(11, 7))

    # Real Exp 2 data
    ax.scatter([100], [0.92], s=400, color=SMOOTH_COLOR, edgecolor="k",
               linewidths=2, zorder=10,
               label="bs=256  (Exp 2): λ=103, robust=0.084")
    ax.scatter([9773], [0.991], s=400, color=BRITTLE_COLOR, edgecolor="k",
               linewidths=2, zorder=10,
               label="bs=5000 (Exp 2): λ=9 773, robust=0.009")

    # Hypothetical disagreement regimes
    ax.scatter([200], [0.85], s=320, color="#d6a72c", edgecolor="k",
               linewidths=2, marker="*", zorder=10,
               label="flat-but-brittle (saturation + LS regime)")
    ax.scatter([5000], [0.25], s=320, color="purple", edgecolor="k",
               linewidths=2, marker="*", zorder=10,
               label="sharp-but-robust (Andriushchenko 2023 regime)")

    # Diagonal-ish trend line
    xs = np.logspace(1, 4.5, 50)
    ys = 0.10 + 0.85 * (np.log10(xs) - 1) / 3.5
    ax.plot(xs, np.clip(ys, 0, 1), "k--", alpha=0.3,
            label="expected if probes agreed perfectly")

    # Quadrant labels (top-left corner of each, light gray)
    ax.text(11, 0.95, "TOP-LEFT\nlow param λ,\nhigh brittleness\n(disagree)",
            fontsize=8.5, color="gray", alpha=0.8,
            bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="lightgray"))
    ax.text(11, 0.08, "BOTTOM-LEFT\nlow λ, low brittleness\n(robust + flat)",
            fontsize=8.5, color="gray", alpha=0.8,
            bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="lightgray"))
    ax.text(15000, 0.95, "TOP-RIGHT\nhigh λ, high brittleness\n(brittle + sharp)",
            fontsize=8.5, color="gray", alpha=0.8, ha="right",
            bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="lightgray"))
    ax.text(15000, 0.08, "BOTTOM-RIGHT\nhigh λ, low brittleness\n(disagree)",
            fontsize=8.5, color="gray", alpha=0.8, ha="right",
            bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="lightgray"))

    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("parameter-space sharpness  (Hessian λ_max, log scale)  →")
    ax.set_ylabel("input-space brittleness  (1 − robust@ε=8/255)  →")
    ax.set_title("Cross-Probe Agreement Map: when do P5 (parameter) and P3 (input) agree?\n"
                 "Diagonal = agreement;  off-diagonal stars = the interesting regimes",
                 fontsize=12)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(1000, color="gray", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "09_cross_probe.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 09_cross_probe.png")


# ==================================================================
# Figure 10 — probe taxonomy / decision matrix
# ==================================================================
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.97, "Probe Taxonomy — what does each probe measure, and when do we use it?",
            fontsize=14, weight="bold", ha="center")

    # Two columns: parameter-space and input-space
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.46, 0.85, fill=True,
                                facecolor="#e8f0ff", edgecolor="black", linewidth=1.5))
    ax.add_patch(plt.Rectangle((0.52, 0.05), 0.46, 0.85, fill=True,
                                facecolor="#fff0e8", edgecolor="black", linewidth=1.5))

    ax.text(0.25, 0.86, "Parameter-space  (Θ ⊂ R^|θ|)",
            fontsize=13, weight="bold", ha="center")
    ax.text(0.75, 0.86, "Input-space  (X ⊂ R^d)",
            fontsize=13, weight="bold", ha="center")

    # Parameter-space probes
    param_probes = [
        ("P5  Hessian top-k", "gold standard for param sharpness\nLanczos via HVP\n→ λ_max, top-5 eigs", "darkgreen"),
        ("P6  Landscape slice", "filter-normalized 2D contour\nvisualization, NOT measurement\n→ contour figure", "orange"),
        ("P7  Linear interp", "loss along θ_A → θ_B line\nmode connectivity test\n→ loss-vs-α curve", "purple"),
    ]
    y = 0.78
    for name, desc, c in param_probes:
        ax.text(0.04, y, name, fontsize=11.5, weight="bold", color=c)
        ax.text(0.04, y - 0.04, desc, fontsize=9.5, family="monospace",
                verticalalignment="top")
        y -= 0.18

    # Input-space probes
    input_probes = [
        ("P1  Margin distribution", "logit-space confidence\n(NOT input-space sharpness!)\n→ histogram + mean/p10", "darkred"),
        ("P2  Input-grad norm", "‖∇_x ℓ‖_2 averaged\nsoftmax-saturates on conf. models\n→ histogram + mean", "darkred"),
        ("P3  Adversarial ε-curve", "**operational gold standard**\nFGSM + PGD-20\n→ accuracy vs ε curve", "darkgreen"),
        ("P4  Boundary thickness", "mean random-direction flip dist\nslow on CIFAR (needs batching)\n→ thickness scalar", "orange"),
        ("P8  Calibration (ECE)", "probability calibration only\nnot a brittleness probe!\n→ ECE + reliability bins", "navy"),
    ]
    y = 0.78
    for name, desc, c in input_probes:
        ax.text(0.54, y, name, fontsize=11.5, weight="bold", color=c)
        ax.text(0.54, y - 0.04, desc, fontsize=9.5, family="monospace",
                verticalalignment="top")
        y -= 0.13

    # Legend
    ax.text(0.5, 0.02,
            "darkgreen = gold standard   |   orange = caveat   |   darkred = misleading alone",
            fontsize=9, ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "10_taxonomy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved 10_taxonomy.png")


if __name__ == "__main__":
    print(f"Generating figures into {OUT.resolve()}")
    fig_intro_boundary()
    fig_p1_margin()
    fig_p2_input_grad()
    fig_p3_adversarial()
    fig_p4_boundary_thickness()
    fig_p5_hessian()
    fig_p6_landscape()
    fig_p7_linear_interp()
    fig_p8_calibration()
    fig_cross_probe()
    fig_taxonomy()
    print("\nAll figures generated.")
