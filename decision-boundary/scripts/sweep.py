"""Run a sweep of training + probe jobs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from playground.sweep.config_loader import expand_sweep  # noqa: E402
from playground.sweep.executor import run_sweep  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--executor",
        choices=("local-serial", "gpu-serial", "gpu-parallel"),
        default="gpu-parallel",
    )
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    runs = expand_sweep(args.config)
    logging.info("expanded %d runs from %s", len(runs), args.config)
    results = run_sweep(
        runs,
        executor=args.executor,
        runs_root=args.runs_root,
        data_root=args.data_root,
        do_probes=not args.no_probes,
        max_parallel=args.max_parallel,
        device=args.device,
    )
    failed = [rid for rid, rc in results.items() if rc != 0]
    if failed:
        logging.error("FAILED runs: %s", failed)
        raise SystemExit(1)
    logging.info("all %d runs succeeded", len(results))


if __name__ == "__main__":
    main()
