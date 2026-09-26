#!/usr/bin/env bash
# Run once on the Lambda GH200 box, after the project has been synced.
# Idempotent - safe to re-run.
#
# Strategy: Lambda's system Python ships with a working CUDA-enabled PyTorch
# (verified torch 2.7.0 + CUDA 12.8 on aarch64). Using uv's isolated venv would
# pull the CPU-only wheel from PyPI's default index. Instead, we use the system
# Python and `pip install --user` only the missing dependencies.

set -euo pipefail

cd "$(dirname "$0")/.."

# Verify system PyTorch has CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'system torch lacks CUDA'; print('cuda:', torch.cuda.get_device_name(0))"

# Install the extras we need into the user site-packages (don't touch system)
# Note: Lambda's system PyTorch is compiled against NumPy 1.x ABI; installing
# NumPy 2.x causes "Numpy is not available" at runtime inside DataLoader workers.
# Pin NumPy below 2 on the server.
python3 -m pip install --user --upgrade --quiet \
    "numpy<2" \
    "pyyaml>=6.0" \
    "scipy>=1.11,<1.14" \
    "pandas>=2.0,<2.3" \
    "matplotlib>=3.8" \
    "tensorboard>=2.16" \
    "tqdm>=4.66" \
    "pytest>=8.0" \
    "pytest-timeout>=2.3"

# Sanity-check the full import chain
python3 -c "
import torch, numpy, scipy, pandas, matplotlib, yaml, tqdm
import sys
sys.path.insert(0, '.')
from playground.config import RunConfig
from playground.train.trainer import Trainer
from playground.probes.margin import probe_margin
print('imports OK, torch.cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"

echo "Server setup complete."
