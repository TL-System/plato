"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

import logging
import math
from types import SimpleNamespace
from typing import Iterable, Optional

import torch

from plato.callbacks.trainer import TrainerCallback
from plato.clients import simple
from plato.utils import fonts


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


class Client(simple.Client):
    """A federated learning client for AFL."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks: Optional[Iterable] = None,
    ):
        callbacks_list = list(trainer_callbacks) if trainer_callbacks else []
        if not any(
            cb == AFLPreTrainingLossCallback
            or getattr(cb, "__class__", None) == AFLPreTrainingLossCallback
            for cb in callbacks_list
        ):
            callbacks_list.append(AFLPreTrainingLossCallback)

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=callbacks_list,
        )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self._get_pre_training_loss()
        logging.info(
            fonts.colourize(
                f"[Client #{self.client_id}] Pre-training loss value: {loss}"
            )
        )
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

    def calc_valuation(self, num_samples, loss):
        """Calculate the valuation value based on the number of samples and loss value."""
        if loss is None or num_samples is None or num_samples <= 0:
            return 0.0
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation

    def _get_pre_training_loss(self) -> Optional[float]:
        """Retrieve the loss captured before local training, with safe fallbacks."""
        loss = None
        trainer_context = getattr(self.trainer, "context", None)
        if trainer_context is not None:
            loss = trainer_context.state.get("pre_train_loss")

        if loss is not None:
            return loss

        try:
            return self.trainer.run_history.get_latest_metric("train_loss")
        except ValueError:
            logging.warning(
                "[Client #%d] AFL: Unable to obtain loss metric; defaulting to zero.",
                self.client_id,
            )
            return 0.0
