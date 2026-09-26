"""Task 1 (on real Mistral) — watch information HOP forward, layer by layer.  [GPU]

Perturb one source token's embedding, run Mistral twice (clean vs perturbed), and
measure how much each position's residual stream changes after every layer:
    delta[L, pos] = || resid_perturbed[L, pos] - resid_clean[L, pos] ||
The change can only travel along allowed attention edges, so it spreads forward by
~W-1 positions per layer: a wavefront = the receptive field growing with depth.

Forward only (no grad) -> cheap (~16-19GB bf16).

Design note: native W=4096 reaches 4095 tokens in ONE layer, so the hop is invisible
unless seq >> 4096. The hopping is a property of the MASK, not the weights, so we can
shrink the window purely to SEE the mechanism. Use --window 16 for the visible demo,
and --window 4096 --seq 6000 to confirm the real model engages the native window.

NOTE: untested until run on GPU.
Run:  python mistral_propagation.py --window 16 --seq 80
      python mistral_propagation.py --window 4096 --seq 6000
"""
import argparse
import warnings

import torch

warnings.filterwarnings("ignore")
MODEL = "mistralai/Mistral-7B-v0.1"
THRESH = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=16, help="override sliding_window for visibility")
    ap.add_argument("--seq", type=int, default=80)
    ap.add_argument("--src", type=int, default=8, help="source token position to perturb")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    model.config.sliding_window = args.window     # mask is rebuilt from config each forward
    W, src = args.window, args.src

    base = ("The receptive field of a deep network grows with depth because every "
            "layer mixes a local neighborhood and the next layer mixes those mixtures. ")
    ids = tok(base * (args.seq // 16 + 1), return_tensors="pt").input_ids[:, : args.seq].to("cuda")
    n = ids.shape[-1]
    print(f"model={MODEL}  W={W}  seq={n}  perturb src={src}\n")

    emb = model.model.embed_tokens(ids)
    pert = torch.zeros_like(emb)
    g = torch.Generator(device="cuda").manual_seed(0)
    pert[0, src] = torch.randn(emb.shape[-1], generator=g, device="cuda", dtype=emb.dtype) * 5.0

    nL = model.config.num_hidden_layers
    with torch.no_grad():
        clean = model(inputs_embeds=emb, output_hidden_states=True, use_cache=False).hidden_states
        dirty = model(inputs_embeds=emb + pert, output_hidden_states=True, use_cache=False).hidden_states

    import numpy as np
    dmat = np.zeros((nL, n))
    print(f"  {'after layer':>11} | {'reached up to pos':>17} | {'fwd reach':>9} | {'theory src+L*(W-1)':>18}")
    for L in range(1, nL + 1):
        delta = (dirty[L][0] - clean[L][0]).float().norm(dim=-1)
        dmat[L - 1] = delta.cpu().numpy()
        reached = (delta > THRESH).nonzero().flatten()
        front = int(reached.max()) if len(reached) else src
        print(f"  {L:>11} | {front:>17} | {front-src:>9} | {min(src + L*(W-1), n-1):>18}")

    print("\nThe front should advance ~W-1 per layer (structural). This IS context growing with depth.")

    # heatmap: delta[layer, pos] -- the triangular wavefront on real Mistral
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(np.log10(dmat + 1e-9), aspect="auto", cmap="magma", origin="lower",
              extent=[0, n, 1, nL])
    ax.axvline(src, color="cyan", ls="--", lw=1, label=f"source s={src}")
    ax.plot([min(src + L * (W - 1), n - 1) for L in range(1, nL + 1)], range(1, nL + 1),
            "c.-", lw=1, ms=4, label="theory front s+L*(W-1)")
    ax.set_xlabel("token position"); ax.set_ylabel("after layer L")
    ax.set_title(f"Mistral-7B-v0.1 SWA forward pass: perturbation hops +{W-1}/layer (W={W})")
    ax.legend(loc="lower right")
    fig.colorbar(ax.images[0], label="log10 |Δ residual|")
    fig.tight_layout()
    fig.savefig("mistral_propagation.png", dpi=120)
    print("saved -> mistral_propagation.png")


if __name__ == "__main__":
    main()
