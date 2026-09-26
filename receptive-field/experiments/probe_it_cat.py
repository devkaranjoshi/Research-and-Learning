"""Ad-hoc ERF probe: gradient sensitivity of the 'it' token to 'cat' (and all others).

Sentence: "cat sat on mat but it tired"
Question: how much does each earlier token influence the residual-stream representation
of 'it', and where does 'cat' rank? (a coreference-flavored use of ruler D).

Measured across layers, since coreference tends to resolve in mid/late layers.
"""
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_probe import Ruler, attn_rf  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

LAYERS = (0, 3, 6, 9, 11)


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    text = "cat sat on mat but it tired"
    ids = tok(text, return_tensors="pt").input_ids
    toks = [tok.decode([t]) for t in ids[0]]
    pos = {t.strip(): i for i, t in enumerate(toks)}  # last occurrence of each word
    print("tokens:", list(enumerate(toks)))

    it_pos = pos["it"]
    cat_pos = pos["cat"]
    print(f"\ntarget = 'it' @ pos {it_pos};  probing influence of 'cat' @ pos {cat_pos}\n")

    print(f"{'layer':>5} | influence of each source token on 'it' (normalized to max)")
    print("-" * 78)
    header = "      | " + "  ".join(f"{t.strip()[:4]:>5}" for t in toks[: it_pos + 1])
    print(header)
    for L in LAYERS:
        infl = attn_rf(model, ids, it_pos, layer=L, ruler=Ruler.GRAD).influence
        upto = infl[: it_pos + 1]
        norm = upto / upto.max()
        cells = "  ".join(f"{norm[j]:5.2f}" for j in range(it_pos + 1))
        # rank of 'cat' among the non-self source tokens
        order = sorted(range(it_pos), key=lambda j: -upto[j].item())  # exclude 'it' itself
        cat_rank = order.index(cat_pos) + 1
        print(f"{L:>5} | {cells}   | cat={norm[cat_pos]:.3f} rank {cat_rank}/{it_pos}")

    # focus summary at a late layer
    L = 9
    infl = attn_rf(model, ids, it_pos, layer=L, ruler=Ruler.GRAD).influence[: it_pos + 1]
    print(f"\nLayer {L} ranking of influence on 'it' (excluding 'it' itself):")
    for rank, j in enumerate(sorted(range(it_pos), key=lambda j: -infl[j].item()), 1):
        star = "   <== cat" if j == cat_pos else ""
        print(f"  {rank}. {toks[j].strip():<5} (pos {j}): {infl[j]/infl.max():.3f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
