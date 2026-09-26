"""Task 2 — Effective Receptive Field of Mistral-7B-v0.1: actual lookback & dilution. [GPU]

For the last token, at each layer, measure (ruler D = gradient of residual-stream norm
w.r.t. input embeddings) how much each PAST token actually influences it, then quantify:

  - influence-vs-distance curve  -> the DILUTION (how fast influence decays back in time)
  - mean effective lookback      = sum_j p_j * (target - j)        ("avg tokens of lookback")
  - 90%-mass radius              = distance holding 90% of total influence ("useful lookback")
  - threshold reach (1% of max)  = outer edge (compare to theoretical L*W)
  - participation ratio          = 1 / sum_j p_j^2  ("effective # tokens attended")

The story: theoretical reach grows LINEARLY with depth (L*W), but effective lookback
grows SUBLINEARLY because each window-hop averages over ~W tokens and dilutes far
contributions. The gap = the dilution cost of sliding-window attention.

Memory: weights FROZEN (measure, not train), SDPA (no N^2 score matrix), --checkpoint
for 40GB. SWA only engages when seq > sliding_window, so use --seq > 4096 (native) or
--window <small> to expose more hops on the real weights.

Run:  python mistral_erf.py --seq 6000 --layers 0 4 8 16 24 31
      python mistral_erf.py --window 512 --seq 4000           # small window -> more hops, richer dilution
"""
import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
MODEL = "mistralai/Mistral-7B-v0.1"


def build_long_input(tok, n_tokens):
    base = ("In deep learning the receptive field of a unit is the set of input "
            "positions that can influence it. ")
    return tok(base * (n_tokens // 16 + 1), return_tensors="pt").input_ids[:, :n_tokens]


def metrics(infl, target):
    """infl: 1D tensor over source positions 0..target (causal). Returns lookback stats."""
    infl = infl[: target + 1].clamp_min(0).cpu()
    total = infl.sum().item() + 1e-9
    p = infl / total
    dist = torch.arange(target, -1, -1, dtype=infl.dtype)      # distance back: target-j
    mean_lookback = float((p * dist).sum())
    # 90% mass radius: smallest D s.t. influence within distance<=D holds >=90% of mass
    order = torch.argsort(dist)                                # near -> far
    csum = torch.cumsum(infl[order], 0) / total
    r90 = float(dist[order][(csum >= 0.90).nonzero()[0, 0]]) if (csum >= 0.90).any() else float(dist.max())
    nz = infl / (infl.max() + 1e-9)
    hit = (nz > 0.01).nonzero().flatten()
    reach = int(target - hit.min().item()) if len(hit) else 0
    part = float(1.0 / (p.pow(2).sum() + 1e-12))               # participation ratio
    return mean_lookback, r90, reach, part, p.cpu().numpy(), dist.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=6000)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
    ap.add_argument("--window", type=int, default=None, help="override sliding_window (else native 4096)")
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--mask-sink", action="store_true", help="zero influence at position 0 (sink control)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.window is not None:
        model.config.sliding_window = args.window
    if args.checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    W = model.config.sliding_window
    ids = build_long_input(tok, args.seq).to("cuda")
    seq = ids.shape[-1]
    target = seq - 1
    print(f"model={MODEL}  W={W}  seq={seq}  target={target}  "
          f"(SWA {'ENGAGED' if seq > W else 'INACTIVE — raise --seq or lower --window'})\n")

    print(f"  {'layer':>5} | {'mean lookback':>13} | {'90% radius':>10} | {'reach(1%)':>9} | "
          f"{'eff #tok':>8} | {'theory L*W':>10} | {'eff/theory':>10}")
    profiles = {}
    for L in args.layers:
        emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, output_hidden_states=True, use_cache=False)
        resid = out.hidden_states[L + 1][0, target].float()
        resid.norm().backward()
        infl = emb.grad[0].norm(dim=-1).float()
        if args.mask_sink:
            infl[0] = 0.0
        ml, r90, reach, part, p, dist = metrics(infl, target)
        profiles[L] = (dist, p)
        theory = min((L + 1) * W, target)
        print(f"  {L:>5} | {ml:>13.1f} | {r90:>10.0f} | {reach:>9} | {part:>8.1f} | "
              f"{theory:>10} | {reach/max(theory,1):>10.3f}")
        del emb, out, resid, infl
        torch.cuda.empty_cache()

    # --- plots: dilution curve + effective-lookback vs depth -----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for L, (dist, p) in profiles.items():
        order = np.argsort(dist)
        ax.plot(dist[order], np.clip(p[order] / p.max(), 1e-6, None), lw=1, label=f"layer {L}")
    ax.set_yscale("log"); ax.set_xlabel("distance back (tokens)"); ax.set_ylabel("normalized influence (log)")
    ax.set_title(f"Mistral ERF: influence vs distance per layer = DILUTION (W={W})")
    ax.legend(); fig.tight_layout(); fig.savefig("mistral_erf_dilution.png", dpi=120)
    print("saved -> mistral_erf_dilution.png  (and run again with --window 512 to see more hops)")


if __name__ == "__main__":
    main()
