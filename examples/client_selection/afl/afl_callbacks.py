"""
Trainer callbacks used by AFL that rely on PyTorch operations.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

from plato.callbacks.trainer import TrainerCallback


class AFLPreTrainingLossCallback(TrainerCallback):
    """Capture the client's loss before any local updates for valuation."""

    def __init__(self):
        self._recorded = False

    def on_train_run_start(self, trainer, config, **kwargs):
        """Reset state at the beginning of each training run."""
        self._recorded = False
        trainer.context.state.pop("pre_train_loss", None)

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Compute the average loss of the current model before local updates."""
        if self._recorded:
            return

        train_loader = getattr(trainer, "train_loader", None)
        if train_loader is None:
            logging.warning(
                "[Client #%d] AFL: Training data loader not available; "
                "cannot record pre-training loss.",
                trainer.client_id,
            )
            return

        if not self._has_batches(train_loader):
            logging.warning(
                "[Client #%d] AFL: Empty training loader; "
                "pre-training loss defaults to zero.",
                trainer.client_id,
            )
            trainer.context.state["pre_train_loss"] = 0.0
            self._recorded = True
            return

        model = trainer.model
        device = trainer.device

        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for examples, labels in train_loader:
                examples = examples.to(device)
                labels = labels.to(device)
                outputs = model(examples)
                loss_tensor = trainer.loss_strategy.compute_loss(
                    outputs, labels, trainer.context
                )
                batch_size = labels.size(0)
                total_loss += loss_tensor.item() * batch_size
                total_examples += batch_size

        if was_training:
            model.train()

        if total_examples > 0:
            trainer.context.state["pre_train_loss"] = total_loss / total_examples
        else:
            trainer.context.state["pre_train_loss"] = 0.0

        logging.debug(
            "[Client #%d] AFL: Recorded pre-training loss %.6f over %d samples.",
            trainer.client_id,
            trainer.context.state["pre_train_loss"],
            total_examples,
        )

        self._recorded = True

    @staticmethod
    def _has_batches(loader: Iterable) -> bool:
        """Best-effort check that the data loader yields at least one batch."""
        length = None
        if hasattr(loader, "__len__"):
            try:
                length = len(loader)
            except TypeError:
                length = None
        return bool(length) if length is not None else True
