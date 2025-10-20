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
from typing import Iterable, List, Optional

from afl_callbacks import AFLPreTrainingLossCallback

from plato.clients import simple
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy
from plato.utils import fonts


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


def _ensure_pretraining_callback(
    trainer_callbacks: Optional[Iterable],
) -> List:
    """Ensure AFL's pre-training loss callback is present once."""
    callbacks_list = list(trainer_callbacks) if trainer_callbacks else []
    if not any(
        cb == AFLPreTrainingLossCallback
        or getattr(cb, "__class__", None) == AFLPreTrainingLossCallback
        for cb in callbacks_list
    ):
        callbacks_list.append(AFLPreTrainingLossCallback)
    return callbacks_list


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks: Optional[Iterable] = None,
):
    """Build an AFL client configured with valuation hooks."""
    callbacks_list = _ensure_pretraining_callback(trainer_callbacks)

    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=callbacks_list,
    )

    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=AFLReportingStrategy(),
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for previous imports that expected a Client callable.
Client = create_client
