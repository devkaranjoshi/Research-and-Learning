"""T3 — Sanity check for the T0 probe rig (DESIGN.md §5).

If any assertion here fails, the rig is lying and every downstream node inherits
the bug. Run before trusting T1/T5/T9/...

Checks:
  C1 causality      : gradient RF gives ~0 influence to FUTURE positions.
  C2 self-dominant  : the target position influences itself most (residual stream).
  C3 window severs  : a width-2 window at layer 0 zeroes out far positions that
                      full attention reaches.
"""
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_probe import Ruler, attn_rf  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

EPS = 1e-5


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    ids = tok("The cat sat on the mat", return_tensors="pt").input_ids
    seq = ids.shape[-1]
    target = 3  # a mid-sequence target so there ARE future positions to test
    print(f"tokens={[tok.decode([t]) for t in ids[0]]}  seq_len={seq}  target_pos={target}\n")

    # ---- C1 + C2 : gradient RF, full attention, layer 0 --------------------
    g = attn_rf(model, ids, target, layer=0, ruler=Ruler.GRAD).influence
    norm = g / g.max()
    print("C1/C2  gradient influence by position (layer 0, full attn):")
    for j in range(seq):
        marker = "  <- target" if j == target else (" (future)" if j > target else "")
        print(f"   pos {j}: {norm[j]:.4f}{marker}")

    future = g[target + 1 :]
    ok_c1 = bool((future < EPS).all())
    ok_c2 = bool(g[target].item() == g.max().item())
    print(f"\n   C1 causality (future ~0): {'PASS' if ok_c1 else 'FAIL'}  "
          f"(max future={future.max().item():.2e})")
    print(f"   C2 self-dominant       : {'PASS' if ok_c2 else 'FAIL'}\n")

    # ---- C3 : window severs distant positions ------------------------------
    target_far = seq - 1  # last token, so window must drop early positions
    full = attn_rf(model, ids, target_far, layer=0, ruler=Ruler.GRAD).influence
    win2 = attn_rf(model, ids, target_far, layer=0, ruler=Ruler.GRAD, window=2).influence
    print(f"C3  layer-0 RF of last token (pos {target_far}), full vs window=2:")
    for j in range(seq):
        print(f"   pos {j}: full={full[j]/full.max():.4f}   win2={win2[j]/ (win2.max()+EPS):.4f}")
    # window=2 at the last token should only keep {target_far-1, target_far}
    severed = win2[: target_far - 1]
    reachable_full = full[: target_far - 1]
    ok_c3 = bool((severed < EPS).all()) and bool((reachable_full > EPS).any())
    print(f"\n   C3 window severs far positions: {'PASS' if ok_c3 else 'FAIL'}  "
          f"(max severed under win2={severed.max().item():.2e}, "
          f"full reaches them={reachable_full.max().item():.2e})\n")

    all_ok = ok_c1 and ok_c2 and ok_c3
    print("=" * 48)
    print(f"T3 SANITY: {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
