"""T9 — Sliding-window attention as effective-RF approximation (DESIGN.md §3, the thesis).

Question: full attention reaches every past token in ONE layer. A window-`w` model
reaches only `w` per layer — but stacking L windowed layers should grow the effective
reach to ~`w x L` (the CNN receptive-field recurrence reborn). Does it?

We measure the gradient RF (ruler D) of the LAST token at several layers, for full
attention vs two window sizes, and report:
  - effective reach  = farthest distance whose influence > 1% of the max
  - the depth law    : does window reach grow ~linearly with layer index?
  - the gap          : full vs window influence-by-distance (the approximation error)

Outputs a numeric table + a log-y plot to docs/figures/t9_swa_erf.png.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_probe import Ruler, attn_rf  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

THRESH = 0.01  # 1% of max counts as "reached"
LAYERS = (0, 3, 7, 11)
WINDOWS = (None, 16, 4)


def reach(influence: torch.Tensor, target: int) -> int:
    """Farthest distance back from `target` with influence > THRESH * max."""
    norm = influence / influence.max()
    hit = (norm > THRESH).nonzero().flatten()
    return int(target - hit.min().item()) if len(hit) else 0


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    text = (
        "In deep learning the receptive field of a unit is the set of input positions "
        "that can influence it; for convolutions this region is fixed by geometry, but "
        "for attention the connectivity is global in a single layer, so the interesting "
        "structure lives entirely in which positions the model actually relies upon."
    )
    ids = tok(text, return_tensors="pt").input_ids
    seq = ids.shape[-1]
    target = seq - 1
    print(f"seq_len={seq}  target_pos={target} (last token)\n")

    # ---- depth law: effective reach vs layer, per window -------------------
    print("Effective reach (distance back with influence > 1% of max):")
    print(f"  {'layer':>6} | {'full':>6} | {'win=16':>7} | {'win=4':>6} | {'16: w*(L+1)':>12} | {'4: w*(L+1)':>11}")
    table = {w: [] for w in WINDOWS}
    for L in LAYERS:
        row = {}
        for w in WINDOWS:
            r = reach(attn_rf(model, ids, target, layer=L, ruler=Ruler.GRAD, window=w).influence, target)
            row[w] = r
            table[w].append(r)
        print(f"  {L:>6} | {row[None]:>6} | {row[16]:>7} | {row[4]:>6} | "
              f"{16*(L+1):>12} | {4*(L+1):>11}")

    # ---- the gap: influence vs distance at the deepest layer --------------
    L = 11
    fig, ax = plt.subplots(figsize=(8, 5))
    for w in WINDOWS:
        infl = attn_rf(model, ids, target, layer=L, ruler=Ruler.GRAD, window=w).influence
        dist = [target - j for j in range(target + 1)]
        vals = (infl[: target + 1] / infl.max()).clamp_min(1e-6).tolist()
        ax.plot(dist, vals, marker=".", ms=3, lw=1, label=f"window={w}" if w else "full attn")
    ax.set_yscale("log")
    ax.set_xlabel("distance back from target (target_pos - source_pos)")
    ax.set_ylabel("normalized influence (log)")
    ax.set_title(f"T9: full vs windowed effective RF at layer {L} (GPT-2-small)")
    ax.axhline(THRESH, color="grey", ls="--", lw=0.7, label="1% reach threshold")
    ax.legend()
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "t9_swa_erf.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved plot -> {out.relative_to(ROOT)}")

    # ---- verdict on the depth law -----------------------------------------
    grows = all(table[4][i] <= table[4][i + 1] for i in range(len(LAYERS) - 1))
    print(f"\nwindow=4 reach grows monotonically with depth: {grows}")
    print(f"full-attention reach at every layer == target ({target}): "
          f"{all(r == target for r in table[None])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
