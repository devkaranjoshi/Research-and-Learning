"""Minimal, self-contained sliding-window attention — the whole mechanism in ~25 lines.

Mirrors exactly what HF Mistral does, but in one readable function:
  - mask predicate:  attend iff  (kv <= q) AND (kv > q - W)     [masking_utils.py:74,88,121]
  - apply:           scores + mask(-inf outside band) -> softmax -> @ V   [modeling_mistral.py:110-117]

Run prints the attention matrix so you can SEE each query attend to only its last W keys.
"""
import torch
import torch.nn.functional as F


def swa_attention(x, W, n_heads=2):
    """One sliding-window causal self-attention layer. x: [seq, d_model]."""
    seq, d = x.shape
    hd = d // n_heads

    # --- standard Q,K,V projections (random here; real model has learned weights) ---
    torch.manual_seed(0)
    Wq, Wk, Wv = (torch.randn(d, d) / d**0.5 for _ in range(3))
    q = (x @ Wq).view(seq, n_heads, hd).transpose(0, 1)        # [heads, seq, hd]
    k = (x @ Wk).view(seq, n_heads, hd).transpose(0, 1)
    v = (x @ Wv).view(seq, n_heads, hd).transpose(0, 1)

    # --- THE WINDOW: build the additive mask from the two predicates ---
    qi = torch.arange(seq)[:, None]      # query index
    ki = torch.arange(seq)[None, :]      # key index
    allowed = (ki <= qi) & (ki > qi - W)                 # kv<=q AND kv>q-W   <-- this IS swa
    mask = torch.where(allowed, 0.0, float("-inf"))      # 0 inside band, -inf outside

    # --- scaled dot-product attention with the band mask ---
    scores = (q @ k.transpose(-1, -2)) / hd**0.5         # [heads, seq, seq]
    scores = scores + mask                               # broadcast mask over heads
    attn = F.softmax(scores, dim=-1)                     # -inf -> 0 probability
    out = (attn @ v).transpose(0, 1).reshape(seq, d)     # [seq, d_model]
    return out, attn


def main():
    seq, d, W = 10, 8, 3
    x = torch.randn(seq, d)
    out, attn = swa_attention(x, W, n_heads=2)
    print(f"seq={seq}, d_model={d}, window W={W}\n")
    print("Attention matrix, head 0 (rows=query q, cols=key kv); each row sums to 1:")
    print("       " + "".join(f"  kv{j} " for j in range(seq)))
    for qi in range(seq):
        cells = "".join(f"{attn[0, qi, j]:5.2f} " for j in range(seq))
        nz = (attn[0, qi] > 1e-6).sum().item()
        print(f"  q{qi:>2} | {cells}  ({nz} keys attended)")
    print(f"\n-> every query attends to at most W={W} keys (the last W positions), "
          f"the rest are exactly 0 (masked -inf -> softmax 0).")
    print(f"-> output shape {tuple(out.shape)} == input shape: all {seq} tokens processed.")


if __name__ == "__main__":
    raise SystemExit(main())
