"""A federated learning client for gradient leakage attacks."""

import pickle

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy
from plato.config import Config


class DLGTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that appends ground-truth data for leakage validation."""

    async def train(self, context):
        report, outbound_payload = await super().train(context)

        file_path = f"{Config().params['model_path']}/{context.client_id}.pickle"
        with open(file_path, "rb") as handle:
            gt_data, gt_labels, target_grad = pickle.load(handle)

        outbound_payload = (outbound_payload, gt_data, gt_labels, target_grad)
        return report, outbound_payload


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a DLG client that appends ground-truth data to payloads."""
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
        training_strategy=DLGTrainingStrategy(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
