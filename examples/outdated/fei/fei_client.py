"""
A federated learning client for FEI.
"""

import logging
import math
from types import SimpleNamespace

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
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
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FeiLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self.trainer.run_history.get_latest_metric("train_loss")
        logging.info(fonts.colourize(f"[Client #{self.client_id}] Loss value: {loss}"))
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

    def calc_valuation(self, num_samples, loss):
        """Calculate the valuation value based on the number of samples and loss value."""
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
