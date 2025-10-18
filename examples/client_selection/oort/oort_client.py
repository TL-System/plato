"""
A federated learning client using Oort.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

import numpy as np

from plato.clients import simple
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy


class OortReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that attaches Oort's statistical utility."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        run_history = getattr(context.trainer, "run_history", None)
        if run_history is None:
            return report

        train_squared_loss_step = run_history.get_metric_values(
            "train_squared_loss_step"
        )

        num_samples = getattr(report, "num_samples", 0)
        if num_samples > 0 and train_squared_loss_step:
            mean_squared_loss = sum(train_squared_loss_step) / num_samples
            report.statistical_utility = num_samples * np.sqrt(mean_squared_loss)
        else:
            report.statistical_utility = 0.0

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
    """Build an Oort client configured with statistical utility reporting."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=OortReportingStrategy(),
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable.
Client = create_client
