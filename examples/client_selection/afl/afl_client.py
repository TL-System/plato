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
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy
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


class AFLReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that annotates AFL valuation metrics."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        loss = self._get_pre_training_loss(context)
        logging.info(
            fonts.colourize(
                f"[Client #{context.client_id}] Pre-training loss value: {loss}"
            )
        )

        num_samples = getattr(report, "num_samples", None)
        report.valuation = self._calc_valuation(num_samples, loss)
        return report

    @staticmethod
    def _calc_valuation(num_samples, loss):
        """Calculate the valuation value based on the number of samples and loss value."""
        if loss is None or num_samples is None or num_samples <= 0:
            return 0.0
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation

    @staticmethod
    def _get_pre_training_loss(context: ClientContext) -> Optional[float]:
        """Retrieve the loss captured before local training, with safe fallbacks."""
        trainer = context.trainer
        if trainer is None:
            return 0.0

        trainer_context = getattr(trainer, "context", None)
        if trainer_context is not None:
            loss = trainer_context.state.get("pre_train_loss")
            if loss is not None:
                return loss

        if getattr(trainer, "run_history", None) is not None:
            try:
                return trainer.run_history.get_latest_metric("train_loss")
            except ValueError:
                logging.warning(
                    "[Client #%d] AFL: Unable to obtain loss metric; defaulting to zero.",
                    context.client_id,
                )
        else:
            logging.warning(
                "[Client #%d] AFL: Trainer history unavailable; defaulting to zero.",
                context.client_id,
            )

        return 0.0


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
        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=self.training_strategy,
            reporting_strategy=AFLReportingStrategy(),
            communication_strategy=self.communication_strategy,
        )
