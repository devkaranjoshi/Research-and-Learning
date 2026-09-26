"""it -> cat head hunt (ruler A): which (layer, head) points the pronoun at the noun?

Sentence: "cat sat on mat but it tired"
For every (layer, head) we read the attention weight A[query='it', key='cat'].
We also check whether 'cat' is the ARGMAX key for the 'it' query in that head
(i.e. the head's dominant look actually lands on cat, not just a nonzero weight).

Complements the gradient probe (ruler D, probe_it_cat.py): ruler A says where heads
*look*; ruler D said where influence *flows*. The point of T1 is they can disagree.
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

from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    text = "cat sat on mat but it tired"
    ids = tok(text, return_tensors="pt").input_ids
    toks = [tok.decode([t]).strip() for t in ids[0]]
    it_pos, cat_pos = toks.index("it"), toks.index("cat")
    print("tokens:", list(enumerate(toks)))
    print(f"query='it'@{it_pos}  key='cat'@{cat_pos}\n")

    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    n_layer, n_head = model.config.n_layer, model.config.n_head

    # grid[L, H] = attention from 'it' to 'cat'; also track argmax key per head
    grid = torch.zeros(n_layer, n_head)
    hits = []  # (attn_it_cat, layer, head, argmax_token)
    for L in range(n_layer):
        a = out.attentions[L][0]                       # [head, q, k]
        for H in range(n_head):
            row = a[H, it_pos]                          # attention from 'it' over keys 0..it_pos
            grid[L, H] = row[cat_pos]
            argmax_key = int(row.argmax())
            hits.append((row[cat_pos].item(), L, H, toks[argmax_key], argmax_key))

    print("Top 8 heads by attention('it' -> 'cat'):")
    print(f"  {'rank':>4} {'layer':>5} {'head':>4} {'A(it->cat)':>11} {'argmax key from it':>20}")
    for r, (w, L, H, amk, ami) in enumerate(sorted(hits, reverse=True)[:8], 1):
        lands = "  <-- lands on CAT" if ami == cat_pos else ""
        print(f"  {r:>4} {L:>5} {H:>4} {w:>11.3f}   {amk!r:>16} (pos {ami}){lands}")

    # heads whose DOMINANT look from 'it' is exactly 'cat'
    dom = [(w, L, H) for (w, L, H, _, ami) in hits if ami == cat_pos]
    print(f"\nHeads whose argmax look from 'it' IS 'cat': {len(dom)}")
    for w, L, H in sorted(dom, reverse=True):
        print(f"    layer {L} head {H}: A(it->cat)={w:.3f}")

    # show the full look of the single strongest it->cat head
    w, L, H, _, _ = max(hits)
    row = out.attentions[L][0][H, it_pos][: it_pos + 1]
    print(f"\nStrongest it->cat head = L{L}H{H}. Its full attention from 'it':")
    for j in range(it_pos + 1):
        star = "  <== cat" if j == cat_pos else ("  (self)" if j == it_pos else "")
        print(f"    {toks[j]:<5} (pos {j}): {row[j]:.3f}{star}")

    # heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title("Attention from 'it' to 'cat' across all heads (GPT-2-small)")
    fig.colorbar(im, label="A(it -> cat)")
    out_png = ROOT / "docs" / "figures" / "it_cat_head_hunt.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"\nsaved heatmap -> {out_png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
