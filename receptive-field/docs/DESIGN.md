# Receptive Field Playground — Design & Theory

**Created:** 2026-06-23
**Status:** Design committed. No code yet. Two kickoff decisions OPEN (see §6).
**Discipline:** paper-driven (see `PAPERS.md`) · one-dial-per-experiment · 9-field runlog (`runlog/`) · research-mentor mode (user writes code + makes decisions).
**Siblings:** `decision-boundary-playground` (outward), `mech-interp-playground` (inward). This sandbox studies **what each unit can see**.

---

## 0. The one question

> *Which input positions can actually influence this one output unit — and how much?*

Two answers, and the gap between them is the whole project:

- **Theoretical RF (TRF)** — what the architecture *allows* (computable by arithmetic).
- **Effective RF (ERF)** — what the network *actually uses* (a magnitude distribution, measured).

---

## 1. Theory — Part A: Convolutional RF (the geometric baseline)

In a CNN the RF is **geometric and fixed**, wired in before any data arrives.

- **TRF recurrence:** `r_l = r_{l-1} + (k_l − 1)·∏_{i<l} s_i` (Araujo 2019).
- **ERF headline (Luo 2016):** the effective RF grows like **√depth**, occupies only a small *central fraction* of the TRF, and has a roughly **Gaussian** falloff. *Big TRF ≠ uses big context.*
- **Mechanism of the gap:** signal from edge pixels passes through many layers and attenuates — pure geometry + depth decay.
- **Measurement:** set the gradient of one center output unit to `1.0`, backprop to the input; `input.grad.abs()` *is* the ERF. No training required for the static map.

## 2. Theory — Part B: Attention RF (the learned, content-dependent case)

In a transformer, full self-attention connects **every token to every token in one layer**, so:

> **TRF is trivially global** (or bounded by the mask). All the structure lives in the **ERF**, which is now **content-dependent and learned** — it varies per input, per head, per layer.

The project shifts from *"how big is the box"* to *"where does influence actually flow, and what shapes it."*

**Why ERF ≪ TRF here is a *different mechanism* than CNNs:** the model *learns to ignore* most of its global connectivity. Heads specialize into recognizable RF shapes:

| Head type | Learned RF shape | Paper |
|-----------|------------------|-------|
| local band | nearby tokens | Clark 2019 |
| previous-token | exactly position `i−1` | Elhage 2021 |
| attention sink | token 0 / BOS | Xiao 2023 |
| induction | last occurrence of current token | Olsson 2022 |

**The ruler problem (must pick before T0).** Four operationalizations of "attention RF", increasing honesty/cost:

| # | Method | Captures | Blind spot | Paper |
|---|--------|----------|------------|-------|
| A | raw attention weights `A[i,j]` | where it *looks* | ignores value norm, OV, MLP, residual | Vaswani 2017; Clark 2019 |
| B | norm-weighted `‖A[i,j]·v_j‖` | look × magnitude | single-layer | Kobayashi 2020 |
| C | attention rollout / flow | multi-layer w/ residual | linear approx | Abnar & Zuidema 2020 |
| D | gradient / activation patching `∂out_i/∂in_j` | **true causal influence** (CNN-ERF twin) | expensive, noisy | Elhage 2021 |

> ⚠️ **Raw attention weights are a lying ruler** — a head can dump 90% of weight on BOS yet barely affect the output. Recommended stance: **A for visualization + D for ground truth**; the A-vs-D gap is itself a finding.

## 3. Theory — Part C: Sliding-window attention as ERF approximation (project thesis)

The justification for the entire efficient-attention family:

1. Full quadratic attention has global TRF but **local-dominant ERF** → you pay O(n²) for connectivity mostly unused.
2. Restrict each layer to a window `w` → cost **O(n·w)**.
3. Recover reach through **depth**: stacked windowed layers give TRF ≈ **`w × L`** — *the CNN receptive-field recurrence reborn.* **SWA is the transformer pretending to be a CNN.**

**Where the approximation is exact vs. where it lies:**

| Distance | Full attention | SWA + depth | Verdict |
|----------|---------------|-------------|---------|
| `d ≤ w` | direct, single-hop | direct, single-hop | ≈ identical |
| `w < d ≤ w·L` | direct, sharp | multi-hop, blurred & attenuated | degrades |
| `d > w·L` | reachable | structurally zero | fails |

The residual error lives entirely in **sharp, long-range, single-hop dependencies** (induction jumps, attention sinks). Each efficient variant is a *patch for the >w hole*:
- **StreamingLLM** (Xiao 2023): re-inject sink tokens (the window severs token 0).
- **Longformer** (Beltagy 2020): window + dilation + a few **global tokens**.
- **BigBird** (Zaheer 2020): window + **random** + global — recovers full-attention expressivity in theory.
- **Interleaved local/global layers** (e.g. Gemma-2): cheap local hops + occasional global "teleport" layer.

> **Thesis (spine of T9–T11):** *Sliding-window attention is a depth-for-width trade that reconstructs full attention's **effective** receptive field — not its theoretical one — and the residual approximation error lives entirely in sharp, long-range, single-hop dependencies, which the variants patch back in with global / random / sink tokens.*

Note the two distinct approximation philosophies: **approximate the matrix** (Linformer low-rank, Performer kernel) vs **approximate the ERF** (SWA). SWA is firmly the second — which is why **ERF, not attention-weight reconstruction error, is the right evaluation lens.**

---

## 4. Research DAG — Part A: Convolutional RF (foundation)

```
N0 probe rig ──┬── N1 TRF analytic ──┐
               ├── N2 ERF empirical ─┴── N4 TRF-vs-ERF gap ──┬─ N5 depth
               └── N3 sanity (1 conv)                        ├─ N6 kernel
                                                             ├─ N7 stride
                                                             ├─ N8 dilation
                                                             └─ N9 pooling
                          N5..N9 ── N10 ERF during training ── N11 RF vs task
                                  ── N12 architectures ── N14 synthesis
```

| Node | Task | Dial | Primary paper |
|------|------|------|---------------|
| N0 | build `effective_rf()` grad probe + `theoretical_rf()` recurrence | — | Araujo 2019 |
| N1 | analytical TRF recurrence | — | Araujo 2019 |
| N2 | empirical ERF map on deep conv stack | — | Luo 2016 |
| N3 | **sanity:** 1 conv layer ⇒ hard kxk box (fail here before trusting rig) | — | — |
| N4 | overlay TRF vs ERF; confirm √depth Gaussian | — | Luo 2016 |
| N5 | depth sweep (2→32) | # layers | Luo 2016 |
| N6 | kernel size (3,5,7) | kernel | Araujo 2019 |
| N7 | stride 1 vs 2 | stride | Araujo 2019 |
| N8 | dilation 1,2,4 (watch gridding holes) | dilation | Yu & Koltun 2016 |
| N9 | add max/avg/strided pooling | downsample | Araujo 2019 |
| N10 | ERF at epochs 0/1/10/50 (CIFAR-10) | training time | Luo 2016 §4 |
| N11 | two-dot-distance task; accuracy collapses when `d > ERF` | task scale | — |
| N12 | ERF of plain / ResNet / U-Net / dilated | architecture | He 2016; Ronneberger 2015 |
| N14 | synthesis (shared with T14) | — | — |

## 5. Research DAG — Part B: Attention RF (current focus)

```
T0 probe rig ──┬── T1 weights-vs-grad (A vs D)
               ├── T2 causal mask = triangular RF
               ├── T3 sanity (1 head/1 layer)
               └── T4 weight ≠ influence
                        └── T5 per-HEAD RF taxonomy ──┬─ T6 depth (layer)
                                                      ├─ T7 #heads / d_head
                                                      ├─ T8 positional encoding
                                                      ├─ T9 sliding window   ◄ thesis
                                                      ├─ T10 sparse / dilated ◄ thesis
                                                      └─ T11 attention sinks  ◄ thesis
                  T6..T11 ── T12 ERF during training ── T13 effective context len ── T14 CNN⟷Transformer
```

| Node | Task | Dial | Primary paper |
|------|------|------|---------------|
| T0 | probe rig: capture `A` (hooks) **and** `D` (grad/patching); `attn_rf(model,prompt,layer,head,method)` | — | Elhage 2021 |
| T1 | overlay raw-attention RF vs gradient RF; quantify divergence | ruler | Kobayashi 2020 |
| T2 | visualize decoder RF strictly backward-looking | mask | Vaswani 2017 |
| T3 | **sanity:** 1 head/1 layer triangle correct (stop if wrong) | — | — |
| T4 | head dumping mass on BOS but ~0 causal effect | — | Xiao 2023 |
| T5 | **classify each head's RF:** local / prev-token / sink / induction | — | Elhage 2021; Olsson 2022 |
| T6 | RF vs layer depth (early local → late global) | layer index | Vig & Belinkov 2019 |
| T7 | #heads / d_head | head count | Michel 2019 |
| T8 | positional encoding: none / learned / RoPE / ALiBi | pos-enc | Su 2021; Press 2021 |
| T9 | **distance-binned ERF: full vs SWA vs SWA+sink vs SWA+global** | attention pattern | Beltagy 2020 |
| T10 | sparse / dilated / random+global | sparsity pattern | Zaheer 2020 |
| T11 | attention sinks; sever & restore token 0 | sink tokens | Xiao 2023 |
| T12 | per-head RF at init vs trained (watch induction emerge) | training time | Olsson 2022 |
| T13 | needle-at-distance recall; RF→behavior link | needle position | Liu 2023 |
| T14 | synthesis: geometric (conv) vs learned (attention) RF | — | — |

### Flagship experiment (T9) — the thesis, falsifiable

Hold model / depth / data fixed; **dial = attention pattern**. Probe with method D (gradient ERF).

> **H:** influence of source `j` on target `i` is (a) indistinguishable full-vs-SWA for `|i−j| ≤ w`; (b) present-but-attenuated-and-broader for SWA in `w < |i−j| ≤ wL`; (c) exactly zero for SWA beyond `wL`; (d) full-attention ERF shows sharp spikes (sinks, induction) that SWA smears/drops — and re-adding global/sink tokens restores them.

**Plot:** ERF magnitude vs distance (log-y), one curve per variant — the full-vs-SWA gap *is* the approximation error, drawn. Then run a needle-at-distance task (T13) and show accuracy collapses right where the SWA ERF hits zero.

---

## 6. Kickoff decisions (RESOLVED 2026-06-23)

- **D1 — Model source → RESOLVED: pretrained GPT-2-small (TransformerLens).** Gives real induction/sink heads for T5/T11 immediately; dials T6–T9 done as cross-config compares / inference-time ablations. From-scratch trainable net deferred to **T12 only** (training dynamics genuinely needs it). *Rejected: from-scratch-first — would delay the head taxonomy and require training before any measurement.*
- **D2 — Primary ruler → RESOLVED: A + D.** Raw weights (A) for visualization, gradient/patching (D) for causal ground truth. The A-vs-D divergence is itself a finding (T1). B (norm) / C (rollout) optional, only if A and D disagree and we need to localize why. *Rejected: A+B only — keeps it descriptive, not causal.*
- **D3 — SWA isolation → RESOLVED: impose window mask at inference.** Same weights, only the attention mask changes ⇒ the T9 ERF gap is *purely* windowing, no retraining confound. *Rejected: train a windowed net — conflates "windowing" with "weights adapted to windowing."*

Consequence: pretrained GPT-2-small + A&D rulers + inference-time masking reaches **T0→T13 with zero training**. Only T12 needs a trainable net.

## 7. Critical paths

- Part A: `N0 → N3 → (N1‖N2) → N4 →` fan out N5–N9 `→ N10 → N11 → N12 → N14`
- Part B: `T0 → T3 → T1 → T5 →` fan out T6–T11 `→ T12 → T13 → T14`

## 8. Logging protocol

Every probe/sweep/training run → append a 9-field block to `runlog/YYYY-MM-DD.md` (template in `runlog/TEMPLATE.md`). Lead with the one dial that changed; cite the grounding paper in *Hypothesis*; be honest in *Surprises* / *What failed*. New papers land in `PAPERS.md` first.
