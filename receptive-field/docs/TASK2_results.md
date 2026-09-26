# Task 2 — Results: Effective Receptive Field & dilution of Mistral-7B-v0.1

**Date:** 2026-06-27 · **Hardware:** NVIDIA GH200 96 GB · **Model:** `mistralai/Mistral-7B-v0.1` (native SWA, W=4096, 32 layers).
**Method:** `TASK2_methodology.md` (gradient ruler D). **Theory:** `TASK1_mistral_swa.md`. **Raw logs:** `runlog/2026-06-27.md`.

---

## 0. The question

For any token, **how many tokens of *actual* lookback does it have**, and how does sliding-window attention **dilute** that influence as depth grows?

## 1. Headline answer

> A token's **effective** receptive field is a small, exponentially-weighted neighborhood — **~500 tokens** (native W=4096, mean lookback ≈ 3.3k) — far below the **8000** reachable in the prompt and the **131,072** (`32×4096`) theoretically reachable. **Theoretical reach grows linearly with depth (`(L+1)·W`); effective lookback grows sublinearly. The widening gap is the dilution**, caused by each window-hop dividing a far token's influence across ~W tokens (≈ exponential decay with distance).

---

## 2. Native window (W=4096, seq=8000) — the real trained behavior

Gradient ERF of the last token, per layer. `--mask-sink` columns zero the position-0 attention sink.

| layer | mean lookback | 90%-mass radius | eff #tokens | reach(1%) | reach(1%, sink-masked) |
|--:|--:|--:|--:|--:|--:|
| 0 | 490 | 2099 | 5.7 | 13 | 13 |
| 4 | 1401 | 4039 | 235 | 7999 | **1293** |
| 8 | 1919 | 4848 | 393 | 7999 | **4083** |
| 16 | 2789 | 5719 | 513 | 7998 | 7998 |
| 24 | 3073 | 6008 | 467 | 7998 | 7998 |
| 31 | 3376 | 6305 | 500 | 7998 | 7998 |

Figure: `figures/mistral_erf_dilution_W4096.png` — influence vs distance, per layer (log-y).
- **Layer 0: a hard cliff at exactly 4096** (single-layer direct reach = W), then the structural-zero floor.
- **Deeper layers leak past 4096 but with a step-down at the boundary** — within-window influence > beyond-window (the extra-hop penalty). The window leaves a "scar" at d=W even at layer 31.
- **Spike at d≈8000** = position-0 attention sink (Xiao 2023).

**Read:** by layer 8 influence *reaches* the whole prompt, but the **effective field is only ~500 tokens** (participation ratio) centered ~3.3k back. Reachable ≠ used.

---

## 3. Small window (W=512, seq=4000) — exposes the depth-dilution

Native W=4096 caps theoretical reach at the sequence length by layer 1, hiding the depth story. With W=512, `(L+1)·W` grows ~512/layer for ~8 layers, so the linear-theory-vs-sublinear-effective divergence is visible.

| layer | theoretical `(L+1)·W` | effective reach (1%) | **eff/theory** | mean lookback | eff #tokens |
|--:|--:|--:|--:|--:|--:|
| 0 | 512 | 475 | **0.93** | 127 | 16.5 |
| 1 | 1024 | 510 | **0.50** | 147 | 24.7 |
| 2 | 1536 | 655 | 0.43 | 199 | 73.5 |
| 4 | 2560 | 1073 | 0.42 | 278 | 175 |
| 8 | 3999 | 1315 | **0.33** | 334 | 245 |
| 16 | 3999* | 1755 | — | 534 | 547 |
| 31 | 3999* | 2147 | — | 653 | 949 |

*theory capped at seq length from layer 8.

Figure: `figures/mistral_erf_dilution_W512.png` — each layer shows **a hard cliff at exactly `(L+1)·W`** (theoretical reach, matches the `M^L` staircase) **plus an ~exponential decay within** (the dilution). Deeper layers push the cliff out and soften the slope.

**Read:** `eff/theory` falls **0.93 → 0.33** over layers 0→8. At the first layer the model uses ~93% of its single-window reach; by layer 8, theoretical reach has grown 8× but effective reach uses only ~33% of it. **Effective lookback grows sublinearly while theoretical reach grows linearly — the dilution, as one number.**

---

## 4. Sink control (H3)

Re-ran both conditions with `--mask-sink` (zero position-0 influence):
- **W=512:** metrics identical (one token of rounding) — sink negligible.
- **W=4096:** mass-weighted metrics (mean lookback, 90%-radius, participation ratio) move **<3%** → dilution finding **robust**. But **threshold-reach was sink-inflated** at shallow layers (L4 7999→1293, L8 7999→4083). Removing the sink revealed the *genuine* effective reach growing with depth (1293→4083→7998).

**Conclusion:** the control killed a *metric*, not the *finding*. Mass-weighted metrics are trustworthy; threshold-reach needs the sink stripped (the gradient is causally correct — the sink truly influences — but a single far spike inflates max-relative metrics while barely touching mass-weighted ones).

---

## 5. Why dilution happens (mechanism)

To influence a token `d` away, signal hops through `d/W` windows; each hop spreads (averages) its contribution across ~W tokens, so a single far token's influence is divided again at every hop → **≈ exponential decay with distance** (the straight log-y slopes in the figures). More depth = more hops = more division = a longer-but-weaker tail. Depth widens the receptive field but dilutes its edge.

This is the convolutional **effective-RF law** (Luo 2016: effective ≪ theoretical, ~√depth-style attenuation) reproduced for a real, trained sliding-window **transformer** — and it is the precise sense in which SWA "approximates" full attention (DESIGN.md §3): it reconstructs the **local** effective field cheaply, and the long tail it nominally reaches is heavily diluted.

---

## 6. Hypotheses outcome

- **H1 (effective ≪ theoretical, sublinear):** ✅ confirmed — eff/theory 0.93→0.33 with depth; effective field ~500 tokens vs 131k theoretical.
- **H2 (native-trained vs imposed-mask GPT-2):** ⏳ not yet directly compared — both show sublinear saturation; a matched-ratio comparison is future work.
- **H3 (position-0 sink present, must be controlled):** ✅ confirmed — sink present; distributional metrics robust to it; threshold-reach is not.

## 7. Limitations

Gradient (1st-order) influence, not attention weights or full ablation; single prompt, last-token target (not a corpus statistic); bf16 noise floor ~1e-3; threshold-reach saturates/sink-sensitive — prefer mass-weighted metrics.

## 8. Artifacts

```
docs/figures/mistral_erf_dilution_W4096.png    native-window dilution curve
docs/figures/mistral_erf_dilution_W512.png     small-window depth-dilution curve
docs/figures/mistral_propagation.png           (Task 1) forward-pass hop staircase
experiments/gpu/mistral_erf.py                 the probe (ruler D + metrics)
docs/TASK2_methodology.md                       full method
docs/runlog/2026-06-27.md                       all runs incl. sink control
```
