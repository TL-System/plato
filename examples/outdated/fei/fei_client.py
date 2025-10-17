"""
A federated learning client for FEI.
"""

import logging
import math

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy
from plato.utils import fonts


class FeiLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that resets the datasource at the beginning of each episode."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)
        if server_response.get("current_round") != 1:
            return

        context.datasource = None
        owner = context.owner
        if owner is not None:
            owner.datasource = None


class FeiReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that annotates FEI valuation metrics."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        trainer = context.trainer
        if trainer is None or getattr(trainer, "run_history", None) is None:
            report.valuation = 0.0
            return report

        loss = trainer.run_history.get_latest_metric("train_loss")
        logging.info(
            fonts.colourize(f"[Client #{context.client_id}] Loss value: {loss}")
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


class Client(simple.Client):
    """A federated learning client for FEI."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FeiLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=FeiReportingStrategy(),
            communication_strategy=communication_strategy,
        )
