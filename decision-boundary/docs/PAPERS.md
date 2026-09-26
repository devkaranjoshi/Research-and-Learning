# Reference Papers — Decision Boundary Playground

This is the authoritative human-readable reference for the playground. **We go by this approach only.**

────────────────────────────────────────

Dial: batch_size → sharpness
  Paper: Keskar, Mudigere, Nocedal, Smelyanskiy, Tang — "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (ICLR)
  Year: 2017
  Key claim: Large-batch SGD finds sharper minima with worse generalization. The 1D landscape interpolation plot from this paper is the iconic brittleness figure.

────────────────────────────────────────

Dial: SAM optimizer
  Paper: Foret, Kleiner, Mobahi, Neyshabur — "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR)
  Year: 2021
  Key claim: Explicit min-max over a parameter neighborhood → flatter minima → better generalization AND robustness.

────────────────────────────────────────

Dial: lr × batch coupling
  Paper: Goyal et al. — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (arXiv)
  Year: 2017
  Key claim: Linear scaling rule: effective LR ≈ lr / batch_size controls SGD noise. Don't sweep them independently.

────────────────────────────────────────

Dial: lr schedule (cosine)
  Paper: Loshchilov, Hutter — "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR)
  Year: 2017
  Key claim: Cosine schedule, sometimes with warm restarts.

────────────────────────────────────────

Dial: weight_decay (decoupled)
  Paper: Loshchilov, Hutter — "Decoupled Weight Decay Regularization" (ICLR — AdamW paper)
  Year: 2019
  Key claim: Weight decay regularizes implicitly, not the same as L2 penalty inside Adam.

────────────────────────────────────────

Dial: don't-decay-LR-grow-batch
  Paper: Smith, Kindermans, Ying, Le — "Don't Decay the Learning Rate, Increase the Batch Size" (ICLR)
  Year: 2018
  Key claim: Treats batch size and LR as a single coupled dial.

────────────────────────────────────────

Dial: memorization & sharpness
  Paper: Zhang, Bengio, Hardt, Recht, Vinyals — "Understanding deep learning requires rethinking generalization" (ICLR)
  Year: 2017
  Key claim: Networks can memorize random labels → sharp boundaries are easy to create.

────────────────────────────────────────

## On the ordering of impact

The "batch_size > optimizer > lr > weight_decay > ..." ordering I gave is practitioner wisdom, not a published result. It comes from:

- Keskar 2017 (cited above) made the batch-size effect by far the most-studied dial.
- Foret 2021 (SAM) created the "switch to SAM gives +X robustness with no other changes" canon.
- Wilson, Roelofs, Stern, Srebro, Recht — "The Marginal Value of Adaptive Gradient Methods in Machine Learning" (NeurIPS) 2017 — pushed back on Adam vs SGD-momentum.

## Papers that challenge the canonical ordering (intellectual honesty)

The picture isn't settled. Two recent results worth knowing:

- Andriushchenko, Croce, Müller, Hein, Flammarion — "A Modern Look at the Relationship Between Sharpness and Generalization" (ICML 2023) — argues that parameter-space sharpness is only weakly causally linked to generalization and challenges the Keskar story. They reproduce sharp minima that generalize fine.
- Dinh, Pascanu, Bengio, Bengio — "Sharp Minima Can Generalize For Deep Nets" (ICML 2017) — earlier theoretical pushback: you can reparameterize a flat minimum into a sharp one without changing the function. So "sharpness in parameter space" is not even well-defined invariant.

**Why this matters for your playground:** Notebook 05 (cross-probe agreement) was specifically designed to surface this tension — when Hessian-λ_max (parameter-space sharpness) disagrees with adversarial-accuracy (input-space sharpness) across your runs, you're reproducing the Andriushchenko & Flammarion result empirically.

## Bottom line

For practical experimentation, the canonical ordering still works — start with batch_size, then optimizer, then LR. But know that the theoretical story behind "sharpness causes brittleness" is still actively debated, and your playground is a tool for forming your own opinion rather than rubber-stamping the textbook one.
