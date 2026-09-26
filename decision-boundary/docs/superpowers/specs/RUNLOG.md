# Run log

## 2026-05-13

### Built playground end-to-end (Tasks 0-21)

Local smoke runs:
- LeNet on MNIST → 98% test acc in 2 epochs (CPU).
- Local sweep runner verified with `_smoke_sweep.yaml` → 2/2 runs OK.

### Task 22 — GH200 smoke sweep (COMPLETED)

End-to-end pipeline verified on Lambda GH200.

**Sweep:** `configs/_smoke_sweep.yaml` (mlp2, MNIST, 1 epoch, batch_size in [64, 128]).
**Executor:** `gpu-parallel --max-parallel 2`.
**Result:** 2/2 runs succeeded; all 4 cheap probes (margin, grad, adversarial, calibration) ran on each.

| Run | test_acc | margin_mean | robust_acc@8/255 | ECE |
|---|---|---|---|---|
| batch_size=64, seed=0 | 0.9500 | 6.794 | 0.916 | 0.0063 |
| batch_size=128, seed=0 | 0.9572 | 6.284 | 0.928 | 0.0038 |

Tidy dataframe via `load_sweep("runs/smoke_sweep")` produces one row per run with all configs + final metrics + probe scalars — verified.

### Deployment quirks discovered + fixed

1. **`uv sync` on aarch64 pulls CPU-only torch wheel.** Lambda's system Python already ships with a working torch 2.7.0 + CUDA 12.8 build. Updated `scripts/server_setup.sh` to use system Python + `pip install --user` for the extras, skipping uv venv on the server.
2. **NumPy 2.x is ABI-incompatible with Lambda's system PyTorch.** "Numpy is not available" inside DataLoader workers. Pin `numpy<2` in `server_setup.sh`.
3. **`tar --exclude='data'` matched `playground/data/` too.** Fixed to path-anchored `--exclude='./data'` in `scripts/sync.sh`.

### Workflow to repeat this run

```bash
# Local (Windows or Linux)
bash scripts/sync.sh push     # tar+ssh push to remote
bash scripts/sync.sh setup    # remote: install deps into system Python user site
ssh -i <key> ubuntu@<ip>      # then:
cd /home/ubuntu/decision-boundary-playground
python3 scripts/sweep.py --config configs/<name>.yaml --executor gpu-parallel --max-parallel 4 --data-root ./data
# back local:
bash scripts/sync.sh fetch
jupyter lab notebooks/        # open 04_sweep_summary.ipynb, set EXPERIMENT
```
