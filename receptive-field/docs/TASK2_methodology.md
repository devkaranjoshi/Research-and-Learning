# Task 2 — Methodology: how we measure the Effective Receptive Field (ERF)

**Goal.** For any output token, determine *how many tokens of actual lookback it has* — and how sliding-window attention *dilutes* that influence as depth grows. We measure this on `mistralai/Mistral-7B-v0.1` (native SWA, W=4096).

**Code:** `experiments/gpu/mistral_erf.py` (ruler D). Probe origin: `src/rf_probe.py`. Theory it tests: `TASK1_mistral_swa.md`, `DESIGN.md §3`.

---

## 1. Definition — what "effective receptive field" means here

The receptive field of an output unit is *which input positions influence it, and how much*. We operationalize "influence" as the **gradient magnitude** of that unit with respect to each input token:

> **ERF(j) = ‖ ∂ y / ∂ e_j ‖** — how much the output unit `y` changes when input token `j`'s embedding `e_j` is perturbed.

This is "ruler D" (the causal/gradient ruler). Rationale: the gradient is the first-order causal sensitivity of the unit to each input — large gradient ⇒ that token shapes the unit; zero gradient ⇒ that token is outside the effective field. It accounts for *everything* on the path (attention weights × values, MLPs, residual stream, RoPE), unlike raw attention weights (ruler A), which we showed can lie (the position-0 sink false-positive, runlog 2026-06-25).

---

## 2. The procedure (exact, per measurement)

A single measurement is defined by a (layer `L`, target position `t`, window `W`) triple. Steps:

1. **Tokenize** a fixed text passage to length `N` (we use `N=8000` native, `N=4000` for the small-window ablation). The passage must satisfy **`N > W`** or the sliding window never engages (SWA ≡ full attention below the window).
2. **Embed and require gradients on the embeddings**, not the token ids (ids are integers → non-differentiable):
   `e = embed_tokens(ids)` ; `e.requires_grad_(True)` — shape `[N, d_model]`.
3. **Forward pass** with the sliding-window mask active (`config.sliding_window = W`), `output_hidden_states=True`, `use_cache=False`.
4. **Pick the output unit** = the residual-stream vector at layer `L`, position `t`:
   `h = hidden_states[L+1][t]` — shape `[d_model]`. (`hidden_states[0]` = embeddings; `hidden_states[L+1]` = output of decoder block `L`.)
5. **Reduce to a scalar and backpropagate:** `‖h‖₂.backward()`.
6. **Read the influence vector:** `influence_j = ‖ e.grad[j] ‖₂` — the L2 norm over `d_model` of each token's embedding gradient. Shape `[N]`.

Positions the causal/window mask forbids receive **exactly zero** gradient (the mask zeroes them pre-softmax, so no signal flows back) — so "outside the field" is unambiguous, not merely small.

### Two design choices that *define* the measurement
- **(C1) What we backprop *from*:** the **L2 norm of the residual-stream vector at (L, t)**. Why the residual norm: we need a single scalar; the norm is rotation-invariant and captures the total magnitude of that unit's representation. This makes it the closest analog of the CNN "single center unit" probe — an *into-this-layer* RF. (Alternative: backprop the final logit → a *behavioural* RF. We chose the layer-residual version for a clean conv↔attention comparison.)
- **(C2) How we reduce the input gradient:** **L2 over `d_model`** → one influence number per source position. (Same reduction used for the GPT-2 phase, so curves are comparable.)

---

## 3. Metrics derived from the influence vector

Let `p_j = influence_j / Σ influence` (an influence distribution over source positions), and distance `d_j = t − j`.

| metric | formula | interpretation |
|--------|---------|----------------|
| **mean effective lookback** | `Σ p_j · d_j` | influence-weighted average tokens of lookback |
| **90%-mass radius** | smallest `D` with `Σ_{d_j ≤ D} p_j ≥ 0.9` | distance holding 90% of influence — robust "useful lookback" |
| **threshold reach** | `max{ d_j : influence_j > 0.01·max }` | outer edge (compare to theoretical `(L+1)·W`) |
| **participation ratio** | `1 / Σ p_j²` | effective *number* of tokens actually attended |
| **eff/theory** | `reach / min((L+1)·W, t)` | fraction of theoretical reach actually used — the **dilution ratio** |

**Why several metrics:** threshold-reach saturates (a single >1% spike at the far end pins it to N), so it is a poor lone measure. Mean-lookback, 90%-radius, and participation ratio capture the *distribution* (the dilution), which is the phenomenon of interest.

---

## 4. Conditions / dials (one variable at a time)

- **Depth dial:** layers `L ∈ {0,4,8,16,24,31}` (native) / `{0,1,2,4,8,16,31}` (small window). Everything else fixed.
- **Window dial:** native `W=4096, N=8000` (the real trained behavior) **and** imposed `W=512, N=4000` (exposes the depth-dilution: with `W=512`, theoretical reach `(L+1)·W` grows ~512/layer for ~8 layers before capping at `N`, so the linear-theory-vs-sublinear-effective divergence is visible — at `W=4096` theory hits `N` by layer 1 and masks it).
- Imposing a smaller `W` than native is an **inference-time mask ablation** on the trained weights (D3): same weights, only the mask changes — isolating the window's effect.

---

## 5. Controls (guarding against artefacts)

- **Causality / window severance** (validated in the GPT-2 phase, `experiments/t3_sanity.py`): future and out-of-window positions must get *exactly* zero influence. Confirmed (zeros are exact, not rounding).
- **Attention-sink control** (`--mask-sink`): the position-0 token is an attention sink (StreamingLLM, Xiao 2023) and shows an influence spike at the far end. We zero it and re-check that the lookback metrics are not inflated by it. **Lesson (runlog 2026-06-25):** never attribute influence to a token sitting at position 0 without a counterfactual — the gradient is causally correct but a position-0 token carries real sink influence that must not be misread as semantic.

---

## 6. Compute setup — and why it doesn't bias the measurement

- **Model:** bf16, `attn_implementation="sdpa"` (never materializes the `N×N` score matrix; honors the window mask; O(N·W)).
- **Weights frozen** (`requires_grad_(False)` on all parameters) — we measure, never train. This only drops the *weight*-gradient buffers (~14.5 GB); the forward activations needed for the input-gradient are still stored, so the ERF value is identical.
- **`--checkpoint`** (gradient checkpointing) optional for ≤40 GB cards; recompute-in-backward, numerically identical result.
- **Hardware:** NVIDIA GH200 96 GB. Seq>4096 backprop is the memory driver (SWA's win is forward-side; the ERF probe is backward-side).

None of bf16/SDPA/freezing/checkpointing change *which* tokens influence the unit or *how much* — they change only speed/memory.

---

## 7. Caveats / limitations

- This is the **gradient** notion of influence (ruler D), a first-order local sensitivity — not attention weights, and not a full nonlinear ablation. It is the most faithful cheap proxy for causal influence (Elhage 2021).
- **Single prompt, last-token target.** Results are one observation, not a corpus statistic. Generalization needs averaging over prompts/targets (future work).
- **bf16 rounding** sets a noise floor (~1e-3 relative); the structural zeros beyond the mask are exact, but tiny in-field values near the floor are not precise.
- **Threshold-reach saturates** at long context with native W — read the distributional metrics instead.

---

## 8. Reproduce

```bash
# native window (real trained behavior)
python experiments/gpu/mistral_erf.py --seq 8000 --layers 0 4 8 16 24 31
# small window (exposes depth-dilution) + sink control
python experiments/gpu/mistral_erf.py --window 512 --seq 4000 --layers 0 1 2 4 8 16 31
python experiments/gpu/mistral_erf.py --window 512 --seq 4000 --mask-sink
```
Env: `transformers==4.57.3`, `numpy<2`, `pillow>=10`, torch 2.7 + CUDA. Model gated (HF login). Outputs: metric table (stdout) + `mistral_erf_dilution*.png`.

---

## 9. One-paragraph summary of the method

We define a token's receptive field as the **gradient of a chosen layer's residual-stream unit with respect to each input token's embedding** (ruler D), measured on Mistral-7B with its sliding-window mask active. From the resulting per-token influence vector we compute distributional lookback metrics (mean lookback, 90%-mass radius, participation ratio) and compare the effective reach to the theoretical `(L+1)·W` bound. Sweeping the layer (depth dial) and the window size (window dial, one variable at a time), with causality and attention-sink controls, lets us read off both *how much actual lookback* a token has and *how that lookback dilutes relative to the theoretical reach as depth grows*.
