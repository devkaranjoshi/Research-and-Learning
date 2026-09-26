# Task 1 — Mistral Sliding-Window Attention: architecture, theory, implementation, depth

**Phase:** SWA research, Task 1 (theory, done on CPU — pure reading/explaining, no model weights loaded).
**Model:** `mistralai/Mistral-7B-v0.1` (the canonical SWA decoder; later Mistral versions drop SWA).
**Companion:** Task 2 runs the ERF experiment on this same model on Lambda GPU.
**Grounding:** real config (fetched) + installed `transformers==4.57.3` source. Paper: Jiang et al. 2023, *Mistral 7B* (PAPERS.md → Mistral2023, to add).

---

## 0. The real numbers (fetched from the config)

| field | value |
|-------|-------|
| layers (`num_hidden_layers`) | **32** |
| attention heads | 32 |
| key/value heads (`num_key_value_heads`) | **8** (Grouped-Query Attention, 4 q-heads per kv-head) |
| hidden size | 4096 |
| **`sliding_window`** | **4096** |
| `max_position_embeddings` | 32768 |
| position encoding | RoPE (rotary) |
| norm | RMSNorm (pre-norm) |

So: a 32-layer causal decoder where **each token attends to at most the 4096 most recent tokens**, yet the trained context is 32768 — 8× the window. How? Depth (§4).

---

## 1. How Mistral uses the sliding window

For a query at position `q`, attention is allowed to key position `kv` iff **both**:

```
kv <= q              (causal: no future)
kv >  q - 4096       (window: no further back than W=4096)
```

i.e. every token sees a **rolling band of the last `W=4096` tokens** `(q-W, q]`, not the whole past. Tokens closer than 4096 back behave exactly like full causal attention; anything older than 4096 is invisible *to that layer*.

Cost consequence: attention is **O(n·W)** instead of **O(n²)**. For sequences longer than the window (n > 4096) this is linear in `n`, which is the entire point — it makes 32k-token context affordable.

This is the same band we imposed on GPT-2 in the previous phase (`causal & (i−j < W)`), so our earlier ablation was a faithful model of real SWA — the difference now is that Mistral's weights were **trained** with this mask, not had it bolted on at inference.

---

## 2. Theory behind it

**Why throw away the long-range connections?** Because (as our T9 results showed) the *effective* receptive field of full attention is local-dominant — most heads, most of the time, rely on nearby tokens. Full O(n²) attention pays for global connectivity it largely doesn't use. SWA bets that **local window + depth** reconstructs the *effective* RF at a fraction of the cost (DESIGN.md §3).

**Two cost regimes:**
- `n <= W`: identical to full causal attention (window never bites).
- `n > W`: O(n·W) — linear; this is where SWA earns its keep.

**The bet, precisely:** SWA does *not* approximate the full attention *matrix* (that's what Linformer/Performer do). It approximates the *effective receptive field*, relying on the empirical fact that influence is local + composable through depth. Task 2 measures whether the bet holds for a model actually trained this way.

---

## 3. Transformer implementation (the real HF code)

### 3.1 The attention module — `MistralAttention` (`modeling_mistral.py:123`)

Standard MHA with three Mistral specifics:
- **GQA**: `k_proj`/`v_proj` project to only 8 kv-heads (`:136-137`); kv is repeated to 32 q-heads inside the attention interface. Shrinks the KV cache 4×.
- **RoPE** applied to q,k each layer (`:158`).
- The window is passed straight into the attention kernel (`:169-179`):

```python
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    scaling=self.scaling,
    sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama  (:177)
    ...
)
```

The comment in the source literally says *"main diff with Llama"* — strip the sliding window and Mistral's attention **is** Llama's.

### 3.2 The mask — `masking_utils.py`

Mistral picks the mask builder by whether a window is set (`modeling_mistral.py:355`):

```python
mask_function = create_causal_mask if self.config.sliding_window is None \
                else create_sliding_window_causal_mask
```

And the sliding-window mask is just two boolean predicates AND-ed (`masking_utils.py:74, 81-90, 117-121`):

```python
def causal_mask_function(b, h, q_idx, kv_idx):     return kv_idx <= q_idx          # :74
def sliding_window_overlay(W):
    def inner(b, h, q_idx, kv_idx):                return kv_idx > q_idx - W        # :88
    return inner
def sliding_window_causal_mask_function(W):
    return and_masks(sliding_window_overlay(W), causal_mask_function)              # :121
```

Modern transformers vmaps these predicate functions over `(q_idx, kv_idx)` to build the 2D mask (or hands the predicate to FlashAttention/SDPA directly), rather than materializing a 32k×32k matrix — important, because a dense mask at max context would be ~1B booleans per layer.

### 3.3 The rolling-buffer KV cache

Because no query ever needs a key older than `W`, generation only stores the **last `W` keys/values per layer** (a ring buffer), instead of the whole history. This is the *memory* payoff of SWA at inference: KV-cache size is capped at `W` regardless of how long the sequence grows. (Set up via `cache_position` / `past_key_values.update`, `modeling_mistral.py:160-163`.)

---

## 4. How it takes help from depth (the crux)

One layer of SWA reaches `W` tokens back. **Stack `L` layers and the reach compounds**, because each layer can pull in what the layer below already gathered:

```
layer 1: token q knows about (q-W, q]
layer 2: each of those tokens already summarized ITS window -> q now indirectly reaches (q-2W, q]
...
layer L: theoretical receptive field  ~=  L * W
```

This is the **exact convolutional receptive-field recurrence** (`r_l = r_{l-1} + (k-1)`), with kernel `k = W`. For Mistral: `32 × 4096 = 131,072` ≈ the 128k effective context Mistral advertises. Depth is what turns a 4096 window into a 131k reach — SWA is a transformer *behaving like a deep CNN* over tokens.

**But our T9 result is the caveat that makes Task 2 interesting.** On GPT-2 with imposed windows we found the *effective* reach is **far below** the theoretical `L×W`: it grows **sublinearly and saturates**, because each inter-window hop attenuates (the Luo-2016 effective≪theoretical law, reproduced for attention). So Mistral's *real* usable context is expected to be well under 131k, and the `L×W` figure is an upper bound.

```
                 theoretical reach  L*W      (linear in depth)        ← the marketing number
   effective reach (measured)  ~ sublinear, saturating               ← what T9 showed for GPT-2
                                  └── Task 2 measures this for a model TRAINED with SWA
```

---

### 4.1 CPU demo — the mask and the depth-cone, made exact (`experiments/swa_depth_demo.py`)

**(A) The mask** is literally the band `kv ≤ q AND kv > q−W`. For n=10, W=3 each query attends to exactly the last 3 tokens:
```
sliding-window-causal (W=3)        rows = query q, cols = key kv
  q3 .###......     q6 ....###...     q9 .......###
```

**(B) Depth = graph reachability.** Information flow across L SWA layers is *exactly* L-hop reachability in the attention graph. If `M` is the boolean mask (with self-loops), `(M^L)[q,kv]` is True iff token `q` can be influenced by `kv` after L layers. Computed via boolean matrix powers (n=64, W=4, last token):

| layer L | reach_back (M^L) | theory (W−1)·L |
|--------:|-----------------:|---------------:|
| 1 | 3 | 3 |
| 4 | 12 | 12 |
| 8 | 24 | 24 |
| 16 | 48 | 48 |

Reach grows by **exactly W−1 per layer** — the convolution recurrence `r_l = r_{l−1}+(k−1)`, verified to the token. Figures:
- `figures/swa_mask_cone.png` — `M^1,M^2,M^4,M^8`: the receptive-field "cone" widening from a thin diagonal band to nearly the full causal triangle.
- `figures/swa_reach_vs_depth.png` — reach vs depth: a perfectly straight `(W−1)·L` line up to the sequence start.

> **This is the THEORETICAL (upper-bound) receptive field.** It is exact and linear. Task 2 measures the *effective* RF (gradient), which T9 showed bends **below** this line and saturates — the gap between the straight line here and the curve there is the whole point.

## 5. What Task 1 sets up for Task 2 (ERF on Lambda)

The open scientific question, now sharp:

> **Does training *with* a sliding window change the effective receptive field versus merely imposing the window at inference?**

Concretely, Task 2 will:
1. Deploy Mistral-7B-v0.1 to Lambda (≈14GB fp16; CPU laptop can't hold it).
2. Run the gradient ERF probe (ruler D, same as GPT-2 phase) on the last token across layers {0, 8, 16, 24, 31}.
3. Measure **effective reach vs depth** and compare to the `L×W` upper bound — does native-SWA Mistral saturate like imposed-mask GPT-2, or does training push the effective reach closer to `L×W`?
4. Watch for **attention sinks** (we learned the hard way these dominate position 0) — Mistral, like all these models, likely parks mass on early tokens; the sink must be controlled before any reach claim (the lesson from 2026-06-25).

Hypotheses to log:
- H1: Mistral effective reach grows sublinearly with depth and saturates < `L×W` (T9 carries over).
- H2: training-with-SWA yields a *larger* effective reach than imposed-mask GPT-2 at matched depth/window-ratio (the network learns to use the window-depth composition the imposed version never trained for).
- H3: position-0 sink present and must be regressed out (control: move probe target / mask the sink).

---

## 6. One-line summary

Mistral SWA = causal attention restricted to the last `W=4096` keys per layer (`kv ≤ q` AND `kv > q−W`), made global-reaching by **depth** (`reach ≈ L×W = 131k`), cheap by **GQA + rolling KV cache** — and its *effective* reach (Task 2) is expected to fall well short of that theoretical bound, exactly as the convolutional ERF does.
