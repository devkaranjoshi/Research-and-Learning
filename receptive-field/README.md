# Receptive Field Playground

> *Which input positions can actually influence this one output unit — and how much?*

A small research sandbox for measuring the **theoretical** vs **effective receptive field**
of transformer attention, with a focus on **sliding-window attention (SWA)**: how stacking
windowed layers grows reach with depth, and how much of that reach the model actually uses.

Two "rulers" are used throughout:

| Ruler | What it measures |
|-------|------------------|
| **A — weights** | raw attention pattern `A[target, source]` — what a head *looks* at |
| **D — gradient** | `∂‖resid[layer, target]‖ / ∂embed[source]` — causal influence |

## Key findings

- **Structural reach grows by exactly `W − 1` per SWA layer.** On real Mistral-7B-v0.1
  (window shrunk to 16), a single-token perturbation's forward wavefront advances +15 tokens
  per layer, matching `src + L·(W−1)` to the token.
- **Effective reach ≪ theoretical reach.** Gradient-based ERF shows influence decays with
  distance; at native `W = 4096` there is a visible step down at the window boundary, and
  the effective number of contributing tokens stays far below what the architecture allows.
- **Position-0 attention sink.** Heavy attention/gradient on the first token is largely a
  sink effect, not semantic — controls in `experiments/it_cat_*_control.py`.

See [`docs/report.html`](docs/report.html) for the full write-up and [`docs/runlog/`](docs/runlog/)
for the day-by-day experiment log.

## Layout

```
src/rf_probe.py           probe rig: rulers A + D, inference-time SWA mask imposition
experiments/              CPU experiments on GPT-2 small + minimal SWA demos
experiments/gpu/          Mistral-7B-v0.1 experiments (needs a ~24–80 GB GPU)
scripts/                  helper scripts
docs/                     design, theory, methodology, results, figures, run log
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/t3_sanity.py        # sanity-check the probe rig first
python experiments/swa_minimal.py      # SWA in ~25 lines
python experiments/t9_swa_erf.py       # SWA effective RF vs depth (GPT-2 small)
```

GPU experiments (Mistral-7B-v0.1 is gated on Hugging Face — accept the license and
`huggingface-cli login` first):

```bash
python experiments/gpu/mistral_propagation.py --window 16 --seq 200 --src 8
python experiments/gpu/mistral_erf.py --seq 8000 --layers 0 4 8 16 24 31
```

See [`docs/TASK2_cloud_deploy.md`](docs/TASK2_cloud_deploy.md) for GPU memory budgeting.

## Docs

- [`DESIGN.md`](docs/DESIGN.md) — theory and experiment plan
- [`TASK1_mistral_swa.md`](docs/TASK1_mistral_swa.md) — how SWA works in Mistral
- [`TASK2_methodology.md`](docs/TASK2_methodology.md) / [`TASK2_results.md`](docs/TASK2_results.md) — ERF method and results
- [`PAPERS.md`](docs/PAPERS.md) — references
