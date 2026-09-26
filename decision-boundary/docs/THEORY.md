# Theory — Why Batch Size Controls Decision Boundary Sharpness

This document explains the causal chain from batch size to boundary brittleness, grounded in papers from `docs/PAPERS.md`. Written as a reference after Experiments 1-2 confirmed the effect empirically.

---

## The five-step chain

```
 large batch_size
       ↓
 low gradient noise (variance ∝ 1/B)
       ↓
 cold optimizer (effective temperature T ∝ η/B)
       ↓
 converges to sharp (narrow) minima
       ↓
 large Hessian eigenvalues → large weight norms → steep f(x)
       ↓
 tiny input perturbation → large logit change → prediction flips easily
       ↓
 BRITTLE DECISION BOUNDARY
```

---

## Step 1: Gradient noise scales as 1/B

The true gradient is `∇L(θ) = E_x[∇ℓ(θ, x)]`. A mini-batch of size B estimates it as:

```
g_B = (1/B) Σᵢ ∇ℓ(θ, xᵢ)
```

Per-sample gradients have covariance `Σ = Cov_x[∇ℓ(θ, x)]`. By the law of large numbers:

```
Cov[g_B] = Σ / B
```

Doubling B halves the noise. This is just sample-mean variance — not specific to deep learning.

**Our data (Exp 2):** bs=256 had ~195 steps/epoch (50000/256); bs=5000 had ~10 steps/epoch. Each step at bs=256 sees 20× more noise per update than bs=5000.

---

## Step 2: SGD as Langevin dynamics (the thermal analogy)

For small learning rate η, continuous-time SGD obeys the stochastic differential equation:

```
dθ = -∇L(θ) dt + √(2T) dW
```

where `dW` is Brownian motion and the effective temperature is:

```
T_eff = η · tr(Σ) / (2B)  ≈  η / B  (up to constants)
```

(Derivation: Smith & Le 2017, "Don't Decay the Learning Rate, Increase the Batch Size".)

This is the same equation that governs a particle in a potential well at temperature T. The particle (= the model parameters) jiggles around due to thermal noise (= gradient noise). Higher T → more jiggling → can escape shallow traps.

**Key insight:** batch size B appears in the denominator of T. Large batch = cold = low jiggle = gets trapped in whatever well it first finds.

---

## Step 3: At equilibrium — hot optimizers prefer wide basins

The stationary distribution of Langevin dynamics in a potential L(θ) is the Boltzmann distribution:

```
p(θ) ∝ exp(-L(θ) / T_eff)
```

Near a local minimum θ*, the loss is locally quadratic:

```
L(θ) ≈ L(θ*) + ½ (θ - θ*)ᵀ H (θ - θ*)
```

where H is the Hessian at θ*. The Boltzmann distribution becomes a Gaussian with covariance `T · H⁻¹`. The probability mass (volume) in a basin scales as:

```
V_basin ∝ T^(d/2) / √det(H)
```

- **Flat minimum** (small H eigenvalues): large det(H)⁻¹/² → large V → attracts the optimizer at any T.
- **Sharp minimum** (large H eigenvalues): small V → only attractive at low T (cold optimizer stays because it can't jiggle out).

At **high temperature** (small batch):
- The optimizer has enough kinetic energy to escape sharp minima.
- It preferentially occupies the flattest minimum accessible — the one with the largest basin.

At **low temperature** (large batch):
- The optimizer gets trapped in the first local minimum it encounters.
- That minimum is often sharp (many sharp minima exist; fewer flat ones).
- It cannot escape because it lacks the noise to climb the basin walls.

---

## Step 4: Why flat minima generalize (Keskar's bridge to test accuracy)

Train loss and test loss are both empirical averages over different samples. You can model their difference as evaluating the loss at a slightly shifted parameter:

```
L_test(θ*) ≈ L_train(θ*) + ½ δᵀ H δ
```

where δ encodes the train→test distribution shift (finite-sample gap, domain shift, etc.).

- **Flat minimum** (small H): `δᵀ H δ` is small → test loss ≈ train loss → good generalization.
- **Sharp minimum** (large H): the same δ gets amplified → test loss spikes → generalization gap.

**Our data (Exp 2):** bs=5000 had λ_max = 9 773 (sharp) and a 4.7-point generalization gap. bs=256 had λ_max = 103 (flat) and no gap beyond noise.

---

## Step 5: The bridge to input-space brittleness (PGD vulnerability)

The theoretical chain from Hessian sharpness to adversarial brittleness is less clean than Steps 1-4 (which are well-established in the optimization theory literature). The argument goes:

### 5a. Sharp minima correlate with large weight norms

SGD paths to sharp minima typically involve high-magnitude gradients near steep curvature. The optimizer "overshoots" and lands with large-norm parameters. Conversely, flat minima can be reached with moderate gradients, leaving smaller weights.

(Caveat: this is empirical, not guaranteed. Dinh et al. 2017 show you can reparameterize to break this link.)

### 5b. Large weight norms → large input-output Jacobian

For a network f(x) = W_L · σ(W_{L-1} · ... · σ(W_1 · x)):

```
‖∂f/∂x‖ ≤ ∏ᵢ ‖Wᵢ‖
```

(Product of spectral norms of weight matrices.) If each ‖Wᵢ‖ is large, ‖∂f/∂x‖ — the sensitivity of the output to input perturbation — is exponentially large.

### 5c. Large Jacobian → vulnerable to FGSM/PGD

An adversary applies perturbation δ with ‖δ‖∞ ≤ ε. The change in logits is approximately:

```
f(x + δ) - f(x) ≈ (∂f/∂x) · δ
```

If ‖∂f/∂x‖ is large, even a tiny ε flips the logits past the decision boundary. The attack budget needed to flip a prediction scales roughly as:

```
ε_flip ∝ margin(x) / ‖∂f/∂x‖_∞
```

- Flat minimum → smaller ‖W‖ → smaller Jacobian → larger ε_flip → robust.
- Sharp minimum → larger ‖W‖ → larger Jacobian → smaller ε_flip → brittle.

**Our data (Exp 2):** bs=256 retained 8.4% robust accuracy at ε = 8/255; bs=5000 collapsed to 0.9%. The Jacobian argument predicts exactly this — the sharp model has steeper decision surfaces, so the PGD perturbation budget of 8/255 is enough to cross them.

---

## Summary diagram

```
                    batch_size ↑
                         │
              gradient noise ↓ (variance ∝ 1/B)
                         │
              T_eff ↓ (cold optimizer)
                         │
            ┌────────────┴────────────┐
            │                         │
     gets trapped in             can't escape to
     sharp minimum               flat minimum
            │                         │
     large Hessian               (what small-batch
     eigenvalues                  SGD achieves)
            │
     large weight norms
            │
     large ‖∂f/∂x‖
            │
     ε_flip is small
            │
     ┌──────┴──────┐
     │             │
  PGD attack    test-time
  succeeds      brittleness
```

---

## Honest caveats

| Challenge | Source | What it says |
|---|---|---|
| Sharpness is coordinate-dependent | Dinh et al. 2017 (ICML) | Rescale layer L by α, layer L+1 by 1/α: the function is unchanged but Hessian eigenvalues change. "Sharpness" isn't a function-space invariant. |
| λ_max only weakly predicts generalization | Andriushchenko et al. 2023 (ICML) | Across many architectures, r ≈ 0.3 between λ_max and test loss. Other factors (data, augmentation, architecture) often dominate. |
| Infinite-width (NTK) limit kills the effect | Jacot et al. 2018 | In the lazy/kernel regime, SGD becomes deterministic gradient flow. The noise-driven basin-selection mechanism vanishes. The Keskar effect requires finite width and practical batch/dataset size ratios. |
| Adam partially adapts away the effect | Kingma & Ba 2015 | Adam divides by running-mean of squared gradients, partially cancelling the noise-variance change from batch size. This is why Keskar used Adam but still saw the effect — it's attenuated but not eliminated. |
| The Jacobian-to-brittleness link (§5) is empirical | Empirical observation | Steps 1-4 have rigorous proofs. Step 5 (Hessian → weight norm → Jacobian → PGD vulnerability) is a correlation observed in practice, not a proven chain. Models exist that break links 5a or 5b. |

---

## What this predicts for our next experiments

| Experiment | Prediction from this theory |
|---|---|
| **Foret 2021 SAM** | SAM's inner maximization explicitly broadens the region of low loss around θ, simulating "high temperature" geometrically rather than stochastically. Predicts: lower λ_max than Adam/SGD at same batch size. |
| **Goyal 2017 linear LR scaling** | Scaling `lr ∝ batch_size` keeps T_eff = η/B constant across batches. Predicts: generalization gap and λ_max ratio disappear when LR is scaled correctly. |
| **Smith 2018 grow-batch** | Increasing batch = decreasing T smoothly. Equivalent to a LR decay schedule. Predicts: same final minimum as cosine LR decay, if the T trajectory matches. |
| **Label smoothing (Müller 2019)** | Caps logit gap → caps weight magnitudes → reduces ‖∂f/∂x‖ directly, independent of the optimization path. Predicts: improved robustness even at large batch size. |

---

## References

Full citations in `docs/REFERENCES.bib`. Key papers for this document:
- Keskar et al. 2017 (batch size → sharpness)
- Smith & Le 2017 (T_eff = η/B derivation)
- Dinh et al. 2017 (reparameterization caveat)
- Andriushchenko et al. 2023 (weak correlation caveat)
- Foret et al. 2021 (SAM — geometric temperature)
- Goyal et al. 2017 (linear scaling rule)
