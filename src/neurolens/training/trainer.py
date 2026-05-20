"""Training loop with W&B + Postgres + JSONL tracking.

Design choices:
- Pure PyTorch loop (no Lightning). Per-batch / per-epoch transparency.
- Optional best-checkpoint save (``save_best_only=True``) — writes to
  ``checkpoint_dir/best.pt`` whenever val accuracy improves.
- Optional early stopping (``early_stopping_patience``).
- Logger is required: every epoch metric goes through ``CompositeLogger``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from neurolens.tracking.composite import CompositeLogger


class Trainer:
    """Stateful trainer wrapping ``model``, ``optimizer``, ``criterion`` and logger.

    Usage::

        trainer = Trainer(model, optimizer, criterion, device, logger)
        trainer.fit(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        logger: CompositeLogger,
        *,
        checkpoint_dir: Path | None = None,
        save_best_only: bool = True,
        early_stopping_patience: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience

        self._best_val_acc: float = -1.0
        self._epochs_since_improvement: int = 0

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one training epoch; log train/loss + train/acc to all sinks."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)

        metrics = {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
        }
        self.logger.log(epoch=epoch, phase="train", metrics=metrics)
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epoch: int,
        phase: str = "val",
    ) -> dict[str, float]:
        """Run one evaluation pass; log <phase>/loss + <phase>/acc."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)
        metrics = {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
        }
        self.logger.log(epoch=epoch, phase=phase, metrics=metrics)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Full train/val loop for ``epochs`` iterations with optional early stop."""
        for epoch in range(epochs):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, epoch, phase="val")

            improved = val_metrics["acc"] > self._best_val_acc
            if improved:
                self._best_val_acc = val_metrics["acc"]
                self._epochs_since_improvement = 0
                self._maybe_save_checkpoint()
            else:
                self._epochs_since_improvement += 1

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"{'*' if improved else ''}"
            )

            if (
                self.early_stopping_patience is not None
                and self._epochs_since_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

    def _maybe_save_checkpoint(self) -> None:
        """Write best.pt to ``checkpoint_dir`` when val accuracy improves."""
        if not self.save_best_only or self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / "best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_val_acc": self._best_val_acc,
            },
            ckpt_path,
        )
