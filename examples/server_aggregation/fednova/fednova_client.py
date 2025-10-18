"""
A federated learning client using FedNova, where the local number of epochs is randomly
generated and communicated to the server at each communication round.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

import logging

import numpy as np

from plato.clients import simple
from plato.clients.strategies.defaults import (
    DefaultLifecycleStrategy,
    DefaultTrainingStrategy,
)
from plato.config import Config


class FedNovaLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that seeds the RNG for each client."""

    def configure(self, context) -> None:
        super().configure(context)
        np.random.seed(3000 + context.client_id)


class FedNovaTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that samples local epochs and annotates reports."""

    async def train(self, context):
        client_id = getattr(context, "client_id", "unknown")

        if (
            hasattr(Config().algorithm, "pattern")
            and Config().algorithm.pattern == "uniform_random"
        ):
            max_local_epochs = getattr(Config().algorithm, "max_local_epochs", 1)
            local_epochs = int(np.random.randint(2, max_local_epochs + 1))
            Config().trainer = Config().trainer._replace(epochs=local_epochs)

            logging.info(
                "[Client #%s] Training with %d epochs.", client_id, local_epochs
            )
        else:
            local_epochs = int(getattr(Config().trainer, "epochs", 1))

        context.state["local_epochs"] = local_epochs

        report, weights = await super().train(context)

        report.epochs = int(local_epochs)

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
    """Build a FedNova client configured with custom lifecycle/training strategies."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=FedNovaLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=FedNovaTrainingStrategy(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
