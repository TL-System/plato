"""
A federated learning client using pruning in Sub-FedAvg.
"""

import logging

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy


class SubFedAvgTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that logs Sub-FedAvg completion."""

    async def train(self, context):
        report, weights = await super().train(context)
        logging.info(
            "[Client #%d] Trained with Sub-FedAvg algorithm.", context.client_id
        )
        return report, weights


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a Sub-FedAvg client that logs each training round."""
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
        training_strategy=SubFedAvgTrainingStrategy(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
