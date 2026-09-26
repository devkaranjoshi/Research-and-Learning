"""Matplotlib plot helpers reused by probes and notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe for headless server use
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_histogram(counts: Sequence[int], edges: Sequence[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    widths = np.diff(edges)
    centers = np.asarray(edges)[:-1] + widths / 2
    ax.bar(centers, counts, width=widths, align="center", edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_landscape(grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cf = ax.contourf(betas, alphas, grid, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title("Filter-normalized loss landscape")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
