"""
A federated learning client of FedSCR.
"""

import logging
from types import SimpleNamespace

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy


class FedSCRLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle hook that updates adaptive thresholds."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)

        trainer = context.trainer
        if trainer is None or not getattr(trainer, "use_adaptive", False):
            return

        thresholds = server_response.get("update_thresholds")
        if thresholds is None:
            return

        threshold = thresholds.get(str(context.client_id))
        if threshold is None:
            return

        trainer.update_threshold = threshold
        logging.info(
            "[Client #%d] Received update threshold %.2f",
            context.client_id,
            threshold,
        )


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

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
            lifecycle_strategy=FedSCRLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wraps up generating the report with any additional information."""
        if self.trainer.use_adaptive:
            report.div_from_global = self.trainer.run_history.get_latest_metric(
                "div_from_global"
            )
            report.avg_update = self.trainer.run_history.get_latest_metric("avg_update")
            report.loss = self.trainer.run_history.get_latest_metric("train_loss")

        return report
