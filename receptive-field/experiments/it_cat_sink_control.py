"""Control: is 'it -> cat' attention coreference, or just the position-0 attention sink?

In the original sentence 'cat' sits at position 0 — GPT-2's first token acts as an
attention sink (Xiao 2023). So heavy 'it -> cat' attention may be the sink, not the
pronoun. Test: move 'cat' OFF position 0 and see whether heads follow the WORD 'cat'
or just POSITION 0.

S0: "cat sat on mat but it tired"               (cat == pos 0)
S1: "the dog saw a cat sat on mat but it tired" (cat is mid-sentence; pos 0 = 'the')
"""
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402


def analyze(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    toks = [tok.decode([t]).strip() for t in ids[0]]
    it_pos = toks.index("it")
    cat_pos = toks.index("cat")
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    nL, nH = model.config.n_layer, model.config.n_head

    follow_cat = follow_pos0 = 0
    a_cat = a_pos0 = 0.0
    for L in range(nL):
        a = out.attentions[L][0]
        for H in range(nH):
            row = a[H, it_pos]
            amax = int(row.argmax())
            follow_cat += amax == cat_pos
            follow_pos0 += amax == 0
            a_cat += row[cat_pos].item()
            a_pos0 += row[0].item()
    tot = nL * nH
    return {
        "text": text, "it_pos": it_pos, "cat_pos": cat_pos,
        "heads_argmax_cat": follow_cat, "heads_argmax_pos0": follow_pos0,
        "mean_A_it_cat": a_cat / tot, "mean_A_it_pos0": a_pos0 / tot, "tot": tot,
        "pos0_tok": toks[0],
    }


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    for text in ["cat sat on mat but it tired",
                 "the dog saw a cat sat on mat but it tired"]:
        r = analyze(model, tok, text)
        same = r["cat_pos"] == 0
        print(f"\n=== {r['text']!r}")
        print(f"    pos0 token = {r['pos0_tok']!r};  cat @ pos {r['cat_pos']}"
              f"{'  (cat IS pos 0)' if same else '  (cat is NOT pos 0)'}")
        print(f"    heads whose argmax look from 'it' = 'cat' : {r['heads_argmax_cat']}/{r['tot']}")
        print(f"    heads whose argmax look from 'it' = pos 0 : {r['heads_argmax_pos0']}/{r['tot']}")
        print(f"    mean A(it->cat)  = {r['mean_A_it_cat']:.3f}")
        print(f"    mean A(it->pos0) = {r['mean_A_it_pos0']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
