"""T0 — Receptive-field probe rig for attention (DESIGN.md §5).

Measures *where influence flows into a target token*, under two rulers:

    A (WEIGHTS) : raw attention pattern   A[target, source]       — what it *looks* at
    D (GRAD)    : gradient influence      ∂out_target / ∂in_source — causal ground truth

Decisions locked 2026-06-23 (DESIGN.md §6), with one tooling amendment:
    - Model    : pretrained GPT-2-small.
    - Tooling  : raw HuggingFace `transformers` + our own hooks (TransformerLens
                 does not install on Python 3.14; HF is already present and is a
                 better fit for the from-scratch-hooks ethos).
    - Rulers   : A + D.
    - SWA (T9) : impose a sliding-window causal mask at INFERENCE by swapping each
                 attention block's `bias` buffer, then restoring it. Never retrain.

Ruler-D design choices (the two that define "RF"; see DESIGN.md §5 T0 note):
    (1) backprop FROM the L2 norm of the residual-stream vector at (layer, target_pos)
        — the closest analog to the CNN "single center unit" probe.
    (2) reduce the input-embedding gradient to one number per position by L2 over d_model.

Papers: Elhage2021 (patching=causal RF), Kobayashi2020 (weights≠influence),
        Beltagy2020 (window reach w×L). See docs/PAPERS.md.

Style: immutable — probes never permanently mutate the model; the window swap is
restored in a finally. Every result is a fresh RFResult.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class Ruler(str, Enum):
    """Which 'ruler' defines the receptive field (DESIGN.md §2 table)."""

    WEIGHTS = "A"   # raw attention weights              (cheap, visual, can lie)
    GRAD = "D"      # ∂out/∂in   — Elhage2021             (causal ground truth)


@dataclass(frozen=True)
class RFResult:
    """One receptive-field measurement (immutable).

    Attributes:
        influence: 1-D tensor [seq_len]; influence of each *source* position on the
            target. Positions the target cannot see are ~0.
        target_pos: Index of the output unit whose RF was measured.
        layer: Layer index probed (0-based block index).
        head: Head index, or None when aggregated over heads (GRAD is always None).
        ruler: Which Ruler produced `influence`.
        window: Sliding-window size imposed at inference, or None for full attention.
    """

    influence: Tensor
    target_pos: int
    layer: int
    head: int | None
    ruler: Ruler
    window: int | None = None


# --------------------------------------------------------------------------- #
# D3 — impose a sliding-window causal mask at inference, then restore it.      #
# --------------------------------------------------------------------------- #
def _windowed_causal(orig_bias: Tensor, window: int) -> Tensor:
    """Intersect GPT-2's lower-triangular `bias` with a band of width `window`.

    position i may attend to j  iff  (j <= i)  and  (i - j < window).
    """
    n = orig_bias.shape[-1]
    idx = torch.arange(n)
    band = (idx.unsqueeze(1) - idx.unsqueeze(0)) < window      # [n, n] bool
    return orig_bias & band.to(orig_bias.device).view(1, 1, n, n)


@contextlib.contextmanager
def _impose_window(model, window: int | None):
    """Temporarily replace every attention block's causal mask with a windowed one."""
    if window is None:
        yield
        return
    saved = [(blk.attn, blk.attn.bias) for blk in model.transformer.h]
    try:
        for attn, bias in saved:
            attn.bias = _windowed_causal(bias, window)
        yield
    finally:
        for attn, bias in saved:
            attn.bias = bias


# --------------------------------------------------------------------------- #
# Public entry point — every downstream node (T1, T3, T5–T13) calls this.      #
# --------------------------------------------------------------------------- #
def attn_rf(
    model,
    input_ids: Tensor,
    target_pos: int,
    *,
    layer: int,
    head: int | None = None,
    ruler: Ruler = Ruler.GRAD,
    window: int | None = None,
) -> RFResult:
    """Measure the receptive field of a single output unit.

    Args:
        model: A HF GPT2LMHeadModel loaded with attn_implementation='eager'.
        input_ids: Token ids, shape [1, seq_len].
        target_pos: Position of the output unit to probe (the 'i' in out_i).
        layer: Block index to read the RF at (0-based).
        head: Head index (WEIGHTS only); None aggregates over heads.
        ruler: WEIGHTS (A) or GRAD (D).
        window: If set, impose a causal sliding window of this size first (T9).

    Returns:
        RFResult whose `.influence` is length seq_len.

    Raises:
        ValueError: If target_pos is out of range.
    """
    seq_len = input_ids.shape[-1]
    if not (0 <= target_pos < seq_len):
        raise ValueError(f"target_pos {target_pos} out of range for seq_len {seq_len}")

    if ruler is Ruler.WEIGHTS:
        influence = _attention_weights(model, input_ids, target_pos, layer, head, window)
    else:
        influence = _gradient_influence(model, input_ids, target_pos, layer, window)
        head = None

    return RFResult(influence.detach().cpu(), target_pos, layer, head, ruler, window)


def _attention_weights(model, input_ids, target_pos, layer, head, window) -> Tensor:
    """Ruler A: the attention row A[target_pos, :] at (layer, head)."""
    with _impose_window(model, window), torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)
    attn = out.attentions[layer][0]                      # [head, q, k]
    row = attn[head, target_pos] if head is not None else attn[:, target_pos].mean(0)
    return row                                           # [seq_len]


def _gradient_influence(model, input_ids, target_pos, layer, window) -> Tensor:
    """Ruler D: L2(∂ ||resid[layer, target_pos]|| / ∂ input_embed[j]) over d_model.

    The residual-stream norm at (layer, target_pos) is the scalar we backprop — the
    transformer twin of the CNN single-center-unit probe.
    """
    emb = model.transformer.wte(input_ids).detach().requires_grad_(True)  # [1, seq, d]
    with _impose_window(model, window):
        out = model(inputs_embeds=emb, output_hidden_states=True)
    # hidden_states: (embeddings, block0_out, block1_out, ...); block `layer` => index layer+1
    resid = out.hidden_states[layer + 1][0, target_pos]                   # [d_model]
    resid.norm().backward()
    return emb.grad[0].norm(dim=-1)                                       # [seq_len]


# --------------------------------------------------------------------------- #
# T9 flagship driver — distance-binned ERF, full vs windowed.                  #
# --------------------------------------------------------------------------- #
def effective_rf_by_distance(
    model,
    input_ids: Tensor,
    target_pos: int,
    *,
    layer: int,
    windows: tuple[int | None, ...] = (None, 8, 2),
    ruler: Ruler = Ruler.GRAD,
) -> dict[int | None, Tensor]:
    """Influence vs source position, one curve per window setting (T9).

    The full-vs-windowed gap *is* the approximation error (DESIGN.md §3).
    """
    return {
        w: attn_rf(model, input_ids, target_pos, layer=layer, ruler=ruler, window=w).influence
        for w in windows
    }
