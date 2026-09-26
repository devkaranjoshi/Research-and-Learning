"""See SWA information-hopping in a REAL forward pass, layer by layer.

We perturb the embedding of ONE source token `s`, run the model twice (clean vs
perturbed) with a sliding window W imposed, and measure how much each position's
residual stream CHANGES after every layer:

    delta[L, pos] = || resid_perturbed[L, pos] - resid_clean[L, pos] ||

Information can only travel along allowed attention edges. With a causal window W,
a perturbation at `s` can reach a forward target `t` only after enough layers for
it to "hop": t <= s + L*(W-1). So delta forms a triangular WAVEFRONT expanding from
s at rate W-1 per layer -- the context literally growing with depth.

Real trained transformer (GPT-2), real weights/nonlinearities; the imposed window
is byte-identical to Mistral's mask (Task 1), so the dynamics are faithful.
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

from rf_probe import _impose_window  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

W = 4            # sliding window
SRC = 4          # source token we perturb
THRESH = 1e-4    # below this = "not reached yet"


def main() -> int:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    text = ("The receptive field of a deep network grows with depth because each "
            "layer mixes information from a small local neighborhood and the next "
            "layer mixes those mixtures together again and again across the stack")
    ids = tok(text, return_tensors="pt").input_ids
    n = ids.shape[-1]
    print(f"seq_len={n}, window W={W}, perturbing source token s={SRC} ('{tok.decode([ids[0,SRC]]).strip()}')\n")

    emb = model.transformer.wte(ids)                          # [1, n, 768]
    pert = torch.zeros_like(emb)
    g = torch.Generator().manual_seed(0)
    pert[0, SRC] = torch.randn(emb.shape[-1], generator=g) * 3.0  # kick one token hard

    with torch.no_grad(), _impose_window(model, W):
        clean = model(inputs_embeds=emb, output_hidden_states=True).hidden_states
        dirty = model(inputs_embeds=emb + pert, output_hidden_states=True).hidden_states

    nL = model.config.n_layer
    delta = torch.zeros(nL, n)
    for L in range(1, nL + 1):                                # hidden_states[L] = output of block L-1
        delta[L - 1] = (dirty[L][0] - clean[L][0]).norm(dim=-1)

    # ---- the wavefront: farthest forward position reached after each layer --
    print(f"  {'after layer':>11} | {'reached up to pos':>17} | {'forward reach':>13} | {'theory s+L*(W-1)':>16}")
    fronts = []
    for L in range(1, nL + 1):
        reached = (delta[L - 1] > THRESH).nonzero().flatten()
        front = int(reached.max()) if len(reached) else SRC
        fronts.append(front)
        print(f"  {L:>11} | {front:>17} | {front - SRC:>13} | {min(SRC + L*(W-1), n-1):>16}")

    # concrete tie-back: when does the perturbation FIRST reach a far target?
    for target in (SRC + 24,):
        if target < n:
            arrives = next((L for L in range(1, nL + 1) if delta[L - 1, target] > THRESH), None)
            print(f"\n  token s={SRC} first affects target t={target} (distance {target-SRC}) "
                  f"at layer {arrives}  (expected ~{(target-SRC)//(W-1)})")

    # ---- heatmap: delta[layer, position] -- the triangular wavefront -------
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow((delta + 1e-9).log10(), aspect="auto", cmap="magma",
                   origin="lower", extent=[0, n, 1, nL])
    ax.axvline(SRC, color="cyan", ls="--", lw=1, label=f"source s={SRC}")
    ax.plot([SRC + L*(W-1) for L in range(1, nL+1)], range(1, nL+1),
            "c.-", lw=1, ms=4, label="theory front s+L*(W-1)")
    ax.set_xlabel("token position"); ax.set_ylabel("after layer L")
    ax.set_title(f"SWA forward pass: perturbation at s={SRC} hops forward W-1={W-1}/layer (GPT-2, W={W})")
    fig.colorbar(im, label="log10 |Δ residual|")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "swa_forward_propagation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
