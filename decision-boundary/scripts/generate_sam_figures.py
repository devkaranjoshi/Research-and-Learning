"""Generate figures for docs/SAM_THEORY.md.

SAM (Sharpness-Aware Minimization, Foret 2021) is best understood through a
1D / 2D loss-surface picture: the two-step ascent-descent, and how the
ascent step makes SAM "feel" curvature. These figures build that intuition.

Run: uv run python scripts/generate_sam_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

SGD_COLOR = "#1f77b4"
SAM_COLOR = "#d62728"
FLAT_COLOR = "#2ca02c"
SHARP_COLOR = "#d62728"


# ==================================================================
# Figure S1 — The SAM two-step on a 1D loss surface
# ==================================================================
def fig_sam_two_step():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    theta = np.linspace(-3, 3, 400)

    # A loss surface with a sharp minimum on the left and a flat one on the right
    def loss(t):
        sharp = 1.6 * np.exp(-((t + 1.4) ** 2) / (2 * 0.12))  # deep narrow well
        flat = 0.9 * np.exp(-((t - 1.2) ** 2) / (2 * 0.9))    # shallow wide well
        return 2.0 - sharp - flat

    L = loss(theta)

    # ---- Panel (a): the ascent step "feels" curvature differently ----
    ax = axes[0]
    ax.plot(theta, L, color="#333", linewidth=2.5, zorder=1)

    rho = 0.35  # neighborhood radius

    for t0, name, c in [(-1.4, "at SHARP min", SHARP_COLOR),
                        (1.2, "at FLAT min", FLAT_COLOR)]:
        # current point
        L0 = loss(t0)
        ax.scatter(t0, L0, s=160, color=c, edgecolor="k", zorder=5)
        # SAM ascent: move by +rho in the direction of increasing loss
        # (numerically pick the side with higher loss)
        gradient_sign = np.sign(loss(t0 + 0.01) - loss(t0 - 0.01))
        t_adv = t0 + gradient_sign * rho
        L_adv = loss(t_adv)
        ax.scatter(t_adv, L_adv, s=160, color=c, marker="^",
                   edgecolor="k", zorder=5)
        ax.annotate("", xy=(t_adv, L_adv), xytext=(t0, L0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        # the "worst-case loss in the rho-ball" = L_adv
        ax.plot([t_adv, t_adv], [L0, L_adv], ":", color=c, alpha=0.6)
        ax.annotate(f"{name}\nworst-case rise\nΔL = {L_adv - L0:.2f}",
                    xy=(t_adv, (L0 + L_adv) / 2),
                    xytext=(t_adv + (0.3 if t0 > 0 else -1.6),
                            (L0 + L_adv) / 2 + 0.25),
                    fontsize=10, color=c,
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor=c, alpha=0.95))

    ax.set_title("(a) The ascent step ε(w) = ρ·g/‖g‖ measures local sharpness\n"
                 "Same ρ, but worst-case loss rise is huge at sharp min, tiny at flat min",
                 fontsize=11)
    ax.set_xlabel("parameter θ")
    ax.set_ylabel("loss L(θ)")
    ax.grid(alpha=0.3)

    # ---- Panel (b): SAM gradient = raw gradient + curvature penalty ----
    # Compare the slope (gradient) at θ vs at the ascended point θ+ε.
    # The DIFFERENCE between these two slopes is the curvature (Hessian) term.
    ax = axes[1]
    ax.plot(theta, L, color="#333", linewidth=2.5, zorder=1)

    t0 = -1.7  # on the steep wall of the sharp well
    L0 = loss(t0)
    grad0 = (loss(t0 + 0.01) - loss(t0 - 0.01)) / 0.02
    grad_sign = np.sign(grad0)
    t_adv = t0 + grad_sign * rho
    L_adv = loss(t_adv)
    grad_adv = (loss(t_adv + 0.01) - loss(t_adv - 0.01)) / 0.02

    # current point + its raw gradient (tangent line)
    ax.scatter(t0, L0, s=180, color=SGD_COLOR, edgecolor="k", zorder=5,
               label="θ (current)")
    tline = np.linspace(t0 - 0.5, t0 + 0.5, 2)
    ax.plot(tline, L0 + grad0 * (tline - t0), color=SGD_COLOR, lw=2,
            linestyle="-", alpha=0.8)
    ax.text(t0 - 0.55, L0 + grad0 * (-0.5) + 0.05,
            f"SGD gradient\nslope={grad0:.1f}", color=SGD_COLOR, fontsize=9,
            ha="right")

    # ascended point + its (steeper) gradient
    ax.scatter(t_adv, L_adv, s=180, color=SAM_COLOR, marker="^",
               edgecolor="k", zorder=5, label="θ + ε(w) (worst-case neighbor)")
    ax.annotate("", xy=(t_adv, L_adv), xytext=(t0, L0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2,
                                linestyle=":"))
    tline2 = np.linspace(t_adv - 0.5, t_adv + 0.5, 2)
    ax.plot(tline2, L_adv + grad_adv * (tline2 - t_adv), color=SAM_COLOR, lw=2,
            alpha=0.8)
    ax.text(t_adv + 0.05, L_adv + 0.15,
            f"SAM uses THIS gradient\nslope={grad_adv:.1f} (steeper)",
            color=SAM_COLOR, fontsize=9)

    ax.set_title("(b) SAM uses the gradient at θ+ε, which is steeper on sharp walls.\n"
                 "First-order:  g_SAM ≈ ∇L(θ) + ρ·H·ĝ  — the extra ρ·H·ĝ term\n"
                 "is the explicit sharpness (Hessian) penalty.", fontsize=10.5)
    ax.set_xlabel("parameter θ")
    ax.set_ylabel("loss L(θ)")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("SAM Two-Step: ascend to worst-case neighbor, then descend using the gradient measured there",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "S1_sam_two_step.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved S1_sam_two_step.png")


# ==================================================================
# Figure S2 — min-max objective: SGD vs SAM minima selection
# ==================================================================
def fig_sam_objective():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    theta = np.linspace(-3, 3, 500)

    def loss(t):
        sharp = 1.7 * np.exp(-((t + 1.3) ** 2) / (2 * 0.05))
        flat = 1.5 * np.exp(-((t - 1.3) ** 2) / (2 * 0.7))
        return 2.0 - sharp - flat

    L = loss(theta)
    rho = 0.3

    # SAM objective = max loss within rho-ball
    def sam_loss(t):
        out = []
        for ti in t:
            window = np.linspace(ti - rho, ti + rho, 41)
            out.append(loss(window).max())
        return np.array(out)

    L_sam = sam_loss(theta)

    # ---- Panel (a): the two objectives ----
    ax = axes[0]
    ax.plot(theta, L, color=SGD_COLOR, linewidth=2.5,
            label="SGD objective: L(θ)")
    ax.plot(theta, L_sam, color=SAM_COLOR, linewidth=2.5,
            label="SAM objective: max_{‖ε‖≤ρ} L(θ+ε)")

    # mark the two minima
    sharp_min_idx = np.argmin(L[theta < 0])
    sharp_min_t = theta[theta < 0][sharp_min_idx]
    flat_min_idx = np.argmin(L[theta > 0])
    flat_min_t = theta[theta > 0][flat_min_idx]

    ax.scatter(sharp_min_t, loss(sharp_min_t), s=150, color=SGD_COLOR,
               edgecolor="k", zorder=5)
    ax.annotate("SGD's global min\n(SHARP)",
                xy=(sharp_min_t, loss(sharp_min_t)),
                xytext=(sharp_min_t - 1.3, loss(sharp_min_t) + 0.3),
                fontsize=10, color=SGD_COLOR,
                arrowprops=dict(arrowstyle="->", color=SGD_COLOR))

    sam_min_t = theta[np.argmin(L_sam)]
    ax.scatter(sam_min_t, sam_loss([sam_min_t])[0], s=150, color=SAM_COLOR,
               edgecolor="k", marker="*", zorder=5)
    ax.annotate("SAM's global min\n(FLAT)",
                xy=(sam_min_t, sam_loss([sam_min_t])[0]),
                xytext=(sam_min_t + 0.2, sam_loss([sam_min_t])[0] + 0.4),
                fontsize=10, color=SAM_COLOR,
                arrowprops=dict(arrowstyle="->", color=SAM_COLOR))

    ax.set_title("(a) SAM minimizes the WORST-CASE loss in a ρ-ball.\n"
                 "The sharp well rises steeply under ρ-perturbation, so SAM's\n"
                 "objective makes it no longer the global minimum.", fontsize=10.5)
    ax.set_xlabel("parameter θ")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Panel (b): how the sharp well "lifts" under SAM ----
    ax = axes[1]
    ax.plot(theta, L, color="#999", linewidth=1.5, linestyle="--",
            label="raw loss L(θ)", alpha=0.7)
    ax.fill_between(theta, L, L_sam, where=(L_sam > L), color=SAM_COLOR,
                    alpha=0.2, label="penalty SAM adds = sharpness")
    ax.plot(theta, L_sam, color=SAM_COLOR, linewidth=2.5,
            label="SAM objective")
    ax.set_title("(b) The shaded gap = local sharpness penalty.\n"
                 "Sharp regions get a big penalty; flat regions get almost none.",
                 fontsize=10.5)
    ax.set_xlabel("parameter θ")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("SAM's min-max objective reshapes the landscape to disfavor sharp minima",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "S2_sam_objective.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved S2_sam_objective.png")


# ==================================================================
# Figure S3 — Expected Exp 3 outcome mock-up + interpretation guide
# ==================================================================
def fig_sam_expected():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel (a): predicted lambda_max bar chart ----
    ax = axes[0]
    optimizers = ["sgd_momentum", "adam", "sam"]
    # Hypothetical predicted lambda_max (NOT measured — for interpretation only)
    predicted_lambda = [2500, 3200, 600]
    pred_err = [400, 500, 150]
    colors = [SGD_COLOR, "#ff7f0e", SAM_COLOR]
    bars = ax.bar(optimizers, predicted_lambda, yerr=pred_err, capsize=6,
                  color=colors, alpha=0.85, edgecolor="k")
    ax.set_ylabel("Hessian λ_max (predicted)")
    ax.set_title("(a) PREDICTION (not data): SAM should land in a flatter basin\n"
                 "→ lower λ_max than SGD/Adam at the same batch size",
                 fontsize=10.5)
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, predicted_lambda):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, f"~{v}",
                ha="center", fontsize=10)
    ax.text(0.5, 0.92, "HYPOTHETICAL — to be confirmed by Exp 3",
            transform=ax.transAxes, ha="center", fontsize=9,
            style="italic", color="darkred",
            bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="darkred"))

    # ---- Panel (b): predicted PGD curves ----
    ax = axes[1]
    eps = np.array([0, 1, 2, 4, 8, 16]) / 255
    # Hypothetical curves
    sgd_curve = np.array([0.83, 0.55, 0.32, 0.12, 0.03, 0.01])
    adam_curve = np.array([0.82, 0.50, 0.27, 0.09, 0.02, 0.005])
    sam_curve = np.array([0.83, 0.66, 0.48, 0.27, 0.11, 0.04])
    ax.plot(eps, sgd_curve, "o-", color=SGD_COLOR, linewidth=2, label="sgd_momentum")
    ax.plot(eps, adam_curve, "s-", color="#ff7f0e", linewidth=2, label="adam")
    ax.plot(eps, sam_curve, "*-", color=SAM_COLOR, linewidth=2.5,
            markersize=12, label="sam (predicted higher)")
    ax.axvline(8 / 255, color="gray", linestyle=":", alpha=0.6)
    ax.text(8 / 255, 0.7, "headline\nε=8/255", fontsize=8, color="gray")
    ax.set_xlabel("PGD-20 perturbation ε")
    ax.set_ylabel("robust accuracy (predicted)")
    ax.set_title("(b) PREDICTION: SAM's curve sits ABOVE SGD/Adam.\n"
                 "If it doesn't, that falsifies the hypothesis.", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment 3 — what we EXPECT to see (interpretation guide, not measured data)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "S3_sam_expected.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved S3_sam_expected.png")


if __name__ == "__main__":
    print(f"Generating SAM figures into {OUT.resolve()}")
    fig_sam_two_step()
    fig_sam_objective()
    fig_sam_expected()
    print("Done.")
