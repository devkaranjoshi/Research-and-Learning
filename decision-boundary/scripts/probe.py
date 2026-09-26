"""Run probes against a trained checkpoint and save results to <run_dir>/probes/."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from playground.config import RunConfig  # noqa: E402
from playground.data import build_dataloaders  # noqa: E402
from playground.models.registry import build_model  # noqa: E402
from playground.probes.adversarial import probe_adversarial  # noqa: E402
from playground.probes.boundary_thickness import probe_boundary_thickness  # noqa: E402
from playground.probes.calibration import probe_calibration  # noqa: E402
from playground.probes.hessian import probe_hessian_top_eigenvalues  # noqa: E402
from playground.probes.input_grad import probe_input_grad  # noqa: E402
from playground.probes.landscape import probe_loss_landscape_slice  # noqa: E402
from playground.probes.margin import probe_margin  # noqa: E402
from playground.viz.plots import save_curve, save_histogram, save_landscape  # noqa: E402

ALL_PROBES = ("margin", "grad", "adv", "thickness", "hessian", "landscape", "calibration")
CHEAP_PROBES = ("margin", "grad", "adv", "calibration")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe")


def _load_run_config(run_dir: Path) -> RunConfig:
    with (run_dir / "config.yaml").open() as f:
        return RunConfig(**yaml.safe_load(f))


def _load_model(run_dir: Path, cfg: RunConfig, device: str) -> torch.nn.Module:
    model = build_model(cfg.model)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _save_json(path: Path, payload: dict) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating | np.integer):
            return o.item()
        raise TypeError(f"unhandled type {type(o)}")

    path.write_text(json.dumps(payload, indent=2, default=default))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument(
        "--probes",
        default="cheap",
        help="comma-separated probe names, or 'cheap' or 'all'",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = _load_run_config(args.run_dir)
    if args.probes == "cheap":
        names = CHEAP_PROBES
    elif args.probes == "all":
        names = ALL_PROBES
    else:
        names = tuple(x.strip() for x in args.probes.split(","))

    _, test_loader = build_dataloaders(cfg, args.data_root)
    model = _load_model(args.run_dir, cfg, args.device)

    probe_dir = args.run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    if "margin" in names:
        logger.info("running probe: margin")
        out = probe_margin(model, test_loader, device=args.device)
        _save_json(probe_dir / "margin.json", out)
        save_histogram(
            out["histogram"]["counts"],
            out["histogram"]["edges"],
            probe_dir / "margin.png",
            "Margin distribution",
        )

    if "grad" in names:
        logger.info("running probe: input_grad")
        out = probe_input_grad(model, test_loader, device=args.device)
        _save_json(probe_dir / "grad.json", out)
        save_histogram(
            out["histogram"]["counts"],
            out["histogram"]["edges"],
            probe_dir / "grad.png",
            "Input grad norm",
        )

    if "adv" in names:
        logger.info("running probe: adversarial (PGD)")
        out = probe_adversarial(model, test_loader, device=args.device, attack="pgd")
        _save_json(probe_dir / "adversarial.json", out)
        save_curve(
            out["epsilons"],
            out["accuracy_by_eps"],
            probe_dir / "adversarial.png",
            "Adversarial accuracy vs eps",
            "eps",
            "accuracy",
        )

    if "calibration" in names:
        logger.info("running probe: calibration")
        out = probe_calibration(model, test_loader, device=args.device)
        _save_json(probe_dir / "calibration.json", out)

    if "thickness" in names:
        logger.info("running probe: boundary thickness")
        out = probe_boundary_thickness(model, test_loader, device=args.device)
        _save_json(probe_dir / "thickness.json", out)

    if "hessian" in names:
        logger.info("running probe: hessian top-k")
        out = probe_hessian_top_eigenvalues(model, test_loader, device=args.device)
        _save_json(probe_dir / "hessian.json", out)

    if "landscape" in names:
        logger.info("running probe: landscape slice")
        out = probe_loss_landscape_slice(model, test_loader, device=args.device)
        np.savez(
            probe_dir / "landscape.npz",
            **{k: v for k, v in out.items() if k != "loss_grid"},
            loss_grid=out["loss_grid"],
        )
        save_landscape(
            out["loss_grid"],
            np.asarray(out["alphas"]),
            np.asarray(out["betas"]),
            probe_dir / "landscape.png",
        )

    logger.info("probes complete: %s", names)


if __name__ == "__main__":
    main()
