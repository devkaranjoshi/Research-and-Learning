"""Train a single run.

Usage:
    python scripts/train.py --config configs/base.yaml --run-dir runs/adhoc/run_000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from playground.config import load_run_config  # noqa: E402
from playground.data import build_dataloaders  # noqa: E402
from playground.models.registry import build_model  # noqa: E402
from playground.train.trainer import Trainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    logger.info("loaded config: model=%s dataset=%s epochs=%d", cfg.model, cfg.dataset, cfg.epochs)

    train_loader, test_loader = build_dataloaders(cfg, args.data_root)
    model = build_model(
        cfg.model,
        num_classes=10,
        width_multiplier=cfg.width_multiplier,
        normalization=cfg.normalization,
        activation=cfg.activation,
        dropout=cfg.dropout,
        dropout_2d=cfg.dropout_2d,
        depth=cfg.depth,
    )
    trainer = Trainer(model, cfg, run_dir=args.run_dir, device=args.device)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()
