"""Control for ruler D: does 'cat's gradient influence on 'it' survive moving cat
off position 0? (i.e. was the earlier grad finding also a position-0 effect?)

Compares grad influence of 'cat' vs the pos-0 token on 'it', for cat-at-0 vs cat-mid.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_probe import Ruler, attn_rf  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

LAYERS = (3, 6, 9)


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    for text in ["cat sat on mat but it tired",
                 "the dog saw a cat sat on mat but it tired"]:
        ids = tok(text, return_tensors="pt").input_ids
        toks = [tok.decode([t]).strip() for t in ids[0]]
        it_pos, cat_pos = toks.index("it"), toks.index("cat")
        print(f"\n=== {text!r}  (cat @ {cat_pos}, pos0={toks[0]!r})")
        for L in LAYERS:
            infl = attn_rf(model, ids, it_pos, layer=L, ruler=Ruler.GRAD).influence[: it_pos + 1]
            norm = infl / infl.max()
            rank = sorted(range(it_pos), key=lambda j: -infl[j].item()).index(cat_pos) + 1
            print(f"   L{L}: grad infl  cat={norm[cat_pos]:.3f} (rank {rank}/{it_pos})   "
                  f"pos0({toks[0]})={norm[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
