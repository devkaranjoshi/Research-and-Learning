"""Training loop with run-folder discipline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from playground.config import RunConfig
from playground.seeding import seed_everything
from playground.train.losses import build_criterion, mixup_batch, mixup_criterion
from playground.train.optimizers import SAM, build_optimizer
from playground.train.schedulers import build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Train a model and persist all artifacts to a single run folder.

    The run folder ends up containing:
        config.yaml          - exact resolved config for the run
        metrics.jsonl        - one JSON object per epoch
        checkpoints/last.pt  - final model + optimizer state
        checkpoints/best.pt  - best test-accuracy checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: RunConfig,
        run_dir: Path,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(cfg.seed)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.criterion = build_criterion(cfg.label_smoothing)

        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    def fit(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        metrics_path = self.run_dir / "metrics.jsonl"
        best_acc = -1.0
        with metrics_path.open("w") as f:
            for epoch in range(self.cfg.epochs):
                train_loss = self._train_one_epoch(train_loader)
                test_loss, test_acc = self._evaluate(test_loader)
                if self.scheduler is not None:
                    self.scheduler.step()
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                if test_acc > best_acc:
                    best_acc = test_acc
                    self._save_checkpoint("best.pt")
                logger.info("epoch %d  train_loss=%.4f  test_acc=%.4f", epoch, train_loss, test_acc)
        self._save_checkpoint("last.pt")

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.cfg.input_perturbation_sigma > 0.0:
                x = x + self.cfg.input_perturbation_sigma * torch.randn_like(x)

            mixed_x, y_a, y_b, lam = mixup_batch(x, y, self.cfg.mixup_alpha)

            if isinstance(self.optimizer, SAM):
                # First forward-backward (compute epsilon(w) ascent)
                self.optimizer.zero_grad()
                loss = mixup_criterion(self.criterion, self.model(mixed_x), y_a, y_b, lam)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # Second forward-backward at ascended weights
                loss2 = mixup_criterion(self.criterion, self.model(mixed_x), y_a, y_b, lam)
                loss2.backward()
                if self.cfg.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                self.optimizer.second_step(zero_grad=True)
                used_loss = loss
            else:
                self.optimizer.zero_grad()
                logits = self.model(mixed_x)
                used_loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                used_loss.backward()
                if self.cfg.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                self.optimizer.step()

            total_loss += used_loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / max(1, n)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            total_loss += self.criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        return total_loss / max(1, n), correct / max(1, n)

    def _save_checkpoint(self, name: str) -> None:
        path = self.run_dir / "checkpoints" / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.cfg),
            },
            path,
        )
