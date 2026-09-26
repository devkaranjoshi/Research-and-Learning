"""Task 1 deep-dive: make SWA's mask and depth-compounding concrete (CPU, no model).

Two exact demonstrations of the Task-1 theory:

  (A) the mask:  a query q attends to key kv iff (kv <= q) AND (kv > q - W).
      We print causal vs sliding-window-causal masks side by side.

  (B) depth gives reach L*W:  information flow across L SWA layers is exactly
      GRAPH REACHABILITY over L hops in the attention graph. If M is the [n,n]
      boolean attention mask (with self-loops), then (M^L)[q, kv] is True iff
      token q can be influenced by token kv after L layers. So:
          theoretical reach_back(q, L) = q - min{ kv : (M^L)[q, kv] }
      This grows by (W-1) per layer until it saturates at the sequence start.

This is the THEORETICAL (upper-bound) receptive field. Task 2 measures the
EFFECTIVE one (gradient), which T9 showed falls well below this line.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def swa_mask(n: int, w: int) -> np.ndarray:
    """Boolean [n, n]: M[q, kv] = (kv <= q) and (kv > q - w)."""
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    return (kv <= q) & (kv > q - w)


def causal_mask(n: int) -> np.ndarray:
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    return kv <= q


def bool_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean matrix product (reachability composition)."""
    return (a.astype(np.int32) @ b.astype(np.int32)) > 0


def reach_back(power: np.ndarray, q: int) -> int:
    """How many tokens back the query q can be influenced by, given M^L."""
    kvs = np.nonzero(power[q])[0]
    return int(q - kvs.min())


def print_mask(name: str, m: np.ndarray) -> None:
    print(f"\n{name}  ('#'=attend, '.'=masked)   rows=query q, cols=key kv")
    print("     " + "".join(f"{j%10}" for j in range(m.shape[1])))
    for q in range(m.shape[0]):
        print(f"  q{q:>2} " + "".join("#" if m[q, j] else "." for j in range(m.shape[1])))


def main() -> int:
    # ---- (A) the mask, small and readable -----------------------------------
    n, w = 10, 3
    print(f"(A) SWA mask predicate:  kv <= q  AND  kv > q - W   (n={n}, W={w})")
    print_mask("causal mask", causal_mask(n))
    print_mask(f"sliding-window-causal mask (W={w})", swa_mask(n, w))
    print(f"\n  -> each query attends to exactly the last W={w} tokens (a band of width {w}).")

    # ---- (B) depth-compounding reach, exact via matrix powers ---------------
    n2, w2 = 64, 4
    M = swa_mask(n2, w2)
    last = n2 - 1
    print(f"\n(B) depth gives reach ~ L*W   (n={n2}, W={w2}, last token = pos {last})")
    print(f"  {'layer L':>7} | {'reach_back (M^L)':>16} | {'theory (W-1)*L':>14} | {'capped@start':>12}")
    powers = []
    P = np.eye(n2, dtype=bool)
    reaches = []
    for L in range(1, 17):
        P = bool_matmul(P, M)
        powers.append((L, P.copy()))
        r = reach_back(P, last)
        reaches.append(r)
        theory = (w2 - 1) * L
        print(f"  {L:>7} | {r:>16} | {theory:>14} | {min(theory, last):>12}")
    print(f"\n  -> reach grows by exactly W-1={w2-1} per layer (the conv recurrence r_l=r_(l-1)+(k-1)),")
    print(f"     until it saturates at the sequence start ({last}). THIS IS THE UPPER BOUND.")

    # ---- figure: the widening RF cone + the reach line ---------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    nc, wc = 32, 4
    Mc = swa_mask(nc, wc)
    Pc = np.eye(nc, dtype=bool)
    show = {1, 2, 4, 8}
    snaps = {}
    Pp = Pc
    for L in range(1, 9):
        Pp = bool_matmul(Pp, Mc)
        if L in show:
            snaps[L] = Pp.copy()
    for ax, L in zip(axes, sorted(show)):
        ax.imshow(snaps[L], cmap="Greys", aspect="equal")
        ax.set_title(f"M^{L}: reach after {L} layers")
        ax.set_xlabel("key kv"); ax.set_ylabel("query q" if L == 1 else "")
    fig.suptitle(f"SWA theoretical receptive field widens with depth (W={wc}) — the 'cone'")
    fig.tight_layout()
    out1 = ROOT / "docs" / "figures" / "swa_mask_cone.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=110)

    fig2, ax = plt.subplots(figsize=(7, 5))
    Ls = list(range(1, 17))
    ax.plot(Ls, reaches, "o-", label=f"actual reach (M^L), W={w2}")
    ax.plot(Ls, [min((w2 - 1) * L, last) for L in Ls], "--", label="(W-1)*L, capped at start")
    ax.axhline(last, color="grey", ls=":", lw=0.8, label=f"sequence start (={last})")
    ax.set_xlabel("number of SWA layers L"); ax.set_ylabel("theoretical reach back (tokens)")
    ax.set_title("Depth turns a width-W window into reach ~ L*W (theoretical / upper bound)")
    ax.legend()
    fig2.tight_layout()
    out2 = ROOT / "docs" / "figures" / "swa_reach_vs_depth.png"
    fig2.savefig(out2, dpi=120)
    print(f"\nsaved -> {out1.relative_to(ROOT)}")
    print(f"saved -> {out2.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
