"""
A federated learning client of FedSCR.
"""

import logging

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy


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


class FedSCRReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that appends FedSCR adaptive metrics."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        trainer = context.trainer
        if trainer is None or not getattr(trainer, "use_adaptive", False):
            return report

        run_history = getattr(trainer, "run_history", None)
        if run_history is None:
            return report

        report.div_from_global = run_history.get_latest_metric("div_from_global")
        report.avg_update = run_history.get_latest_metric("avg_update")
        report.loss = run_history.get_latest_metric("train_loss")
        return report


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a FedSCR client that attaches adaptive thresholds to reports."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=FedSCRLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=FedSCRReportingStrategy(),
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
