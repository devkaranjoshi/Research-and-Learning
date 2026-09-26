"""Sweep executors.

Each run gets its own process so they can share the GPU without cross-contaminating
PyTorch RNG state or autograd caches. We use ``concurrent.futures.ProcessPoolExecutor``
for simplicity - the cost is one process startup per run, which is dwarfed by training.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import yaml

from playground.sweep.config_loader import SweepRun

logger = logging.getLogger(__name__)


def _write_run_config(run: SweepRun, runs_root: Path) -> Path:
    cfg_dir = runs_root / run.config.experiment_name / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{run.run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(asdict(run.config), sort_keys=False))
    return cfg_path


def _run_one(
    run: SweepRun,
    runs_root: Path,
    data_root: Path,
    do_probes: bool,
    device: str,
) -> tuple[str, int]:
    cfg_path = _write_run_config(run, runs_root)
    run_dir = runs_root / run.config.experiment_name / run.run_id
    project_root = Path(__file__).resolve().parent.parent.parent
    train_cmd = [
        sys.executable,
        str(project_root / "scripts" / "train.py"),
        "--config",
        str(cfg_path),
        "--run-dir",
        str(run_dir),
        "--data-root",
        str(data_root),
        "--device",
        device,
    ]
    rc = subprocess.call(train_cmd)
    if rc != 0:
        return run.run_id, rc
    if do_probes:
        # 2026-05-14: skip "thickness" for deep-probe cells until probe_boundary_thickness is
        # batched (the un-batched version hangs on CIFAR + PreActResNet/SmallCNN test loaders).
        probe_names = (
            "margin,grad,adv,calibration,hessian,landscape" if run.deep_probe else "cheap"
        )
        rc = subprocess.call(
            [
                sys.executable,
                str(project_root / "scripts" / "probe.py"),
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--probes",
                probe_names,
                "--device",
                device,
            ]
        )
    return run.run_id, rc


def run_sweep(
    runs: Sequence[SweepRun],
    *,
    executor: str,
    runs_root: Path,
    data_root: Path,
    do_probes: bool,
    max_parallel: int = 4,
    device: str = "cuda",
) -> dict[str, int]:
    """Execute a sweep and return ``{run_id: exit_code}``.

    Executors:
        local-serial : one process, CPU-only (forces device='cpu')
        gpu-serial   : one process at a time, uses 'device'
        gpu-parallel : up to max_parallel processes concurrently, uses 'device'
    """
    if executor == "local-serial":
        device = "cpu"
        max_parallel = 1
    elif executor == "gpu-serial":
        max_parallel = 1
    elif executor != "gpu-parallel":
        raise ValueError(f"unknown executor: {executor}")

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    results: dict[str, int] = {}
    if max_parallel == 1:
        for r in runs:
            rid, rc = _run_one(r, runs_root, data_root, do_probes, device)
            results[rid] = rc
            logger.info("done: %s rc=%d", rid, rc)
        return results

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_run_one, r, runs_root, data_root, do_probes, device) for r in runs]
        for fut in as_completed(futures):
            rid, rc = fut.result()
            results[rid] = rc
            logger.info("done: %s rc=%d", rid, rc)
    return results
