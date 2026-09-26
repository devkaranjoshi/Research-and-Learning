# Task 2 — Mistral-7B-v0.1 on Lambda: deploy plan, GPU budget, run plan

**Model (LOCKED):** `mistralai/Mistral-7B-v0.1` — dense 7B, `sliding_window=4096`, 32 layers, 32 heads / 8 KV (GQA), hidden 4096, RoPE, max_pos 32768. The only Mistral-family model with SWA (v0.2/v0.3 → `sliding_window=None`; Mixtral → MoE, no SWA).

**Why GPU:** local laptop is 17 GB / CPU. 7B needs a GPU. Tasks run sequentially → provision for the heavier one (Task 2 backprop).

---

## 1. GPU memory budget

| Job | precision | peak VRAM | fits |
|-----|-----------|-----------|------|
| Task 1 — forward-only propagation demo (seq ≤ 2k) | bf16 | ~16–18 GB | **1× A10 24GB** ✓ |
| Task 1 — forward-only, long seq ~6k (2 passes, hidden states) | bf16 | ~19–21 GB | **1× A10 24GB** (tight ok) |
| Task 2 — ERF grad probe, seq ~5k (just over window) | bf16, SDPA, weights frozen | ~35–43 GB | A100 40GB (tight) |
| Task 2 — ERF grad probe, seq ~8k | bf16, SDPA, weights frozen | ~45–55 GB | **A100 80GB / H100 80GB** |
| Task 2 — same + gradient checkpointing | bf16 | ~24–30 GB | A100 40GB (≈2× slower) |

**Recommendation — split the work across two instances to save cost:**
- **Task 1 → 1× A10 24 GB** (Lambda's cheapest GPU). Forward-only fits with room; iterate the propagation demo cheaply here.
- **Task 2 → 1× A100 80 GB or H100 80 GB** (single GPU; 7B fits on one card). Spin up only when running the backprop ERF probe. GH200 96GB even more comfortable.

This avoids paying 80GB-card rates while exploring Task 1. (Forward-only Task 1 at seq~6k stores two hidden-state stacks ≈3GB on top of the 14.5GB weights — still under 24GB.)

Two memory levers baked into the scripts:
- **Freeze all weights** (`requires_grad_(False)`); only input embeddings require grad → drops the ~14.5 GB weight-grad buffers (we measure, never train).
- **SDPA attention** (not eager) → never materializes the `[heads × seq²]` score matrix.
- `--checkpoint` flag toggles gradient checkpointing for the 40 GB fallback.

**Memory driver to remember:** SWA only engages when **seq > 4096**. Sub-4096 sequences make SWA ≡ full attention, so Task 2 *must* use long sequences — that is what costs the VRAM.

---

## 2. Deploy steps (Lambda)

1. Spin up a Lambda instance: **A100 80GB** (or H100 80GB).
2. **HF access:** `Mistral-7B-v0.1` is **gated** — accept the license on the model page, then on the instance:
   ```bash
   pip install -U torch transformers accelerate huggingface_hub matplotlib numpy
   huggingface-cli login        # paste HF token with access to the gated repo
   ```
3. **Sync code** (reuse the playground's pattern, `scripts/sync.sh`):
   ```bash
   rsync -av --exclude '.git' receptive-field-playground/ ubuntu@<lambda-ip>:~/rf/
   ```
4. **Pre-download weights** (~14.5 GB bf16) once:
   ```bash
   python -c "from transformers import AutoModelForCausalLM as M; \
              M.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype='bfloat16')"
   ```

---

## 3. Run plan

```bash
# --- Task 1  on 1x A10 24GB (cheap, forward-only) ---
python experiments/gpu/mistral_propagation.py --window 16          # window shrunk for VISIBILITY
python experiments/gpu/mistral_propagation.py --window 4096 --seq 6000   # native window, long seq

# --- Task 2  on 1x A100/H100 80GB (backprop) ---
python experiments/gpu/mistral_erf.py --seq 6000 --layers 0 8 16 24 31
python experiments/gpu/mistral_erf.py --seq 6000 --checkpoint        # 40GB fallback
```

### Design note — the propagation demo needs a *small* window to be visible
Native `W=4096` reaches 4095 tokens forward in a **single** layer, so the wavefront is invisible unless seq ≫ 4096. The structural hopping (front advances W−1 per layer) is a property of the **mask**, not the weights, so we shrink the window (e.g. 16) purely to *see* the mechanism — clearly labelled. A second run at native `W=4096, seq=6000` confirms the real model engages the window (front jumps the full window per layer).

---

## 4. Hypotheses (log against PAPERS.md → Mistral2023, Luo2016)

- **H1:** effective reach grows **sublinearly** with depth and saturates **≪ L×W** (= 131k theoretical) — the T9 result carries to a natively-trained SWA model.
- **H2:** native-SWA Mistral shows a **larger** effective reach at matched depth/window-ratio than imposed-mask GPT-2 (training learns to exploit window-depth composition).
- **H3:** position-0 **attention sink** present (lesson 2026-06-25) — must be controlled before any reach claim; probe both with and without sink-masking, and never attribute influence to a token sitting at position 0 without the move-off-position-0 control.

---

## 5. Status
Scripts written on CPU (untested until GPU): `experiments/gpu/mistral_propagation.py`, `experiments/gpu/mistral_erf.py`. Ready to `rsync` + run once the instance is up.
