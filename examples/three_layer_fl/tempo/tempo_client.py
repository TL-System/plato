"""
A federated learning client of Tempo.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.config import Config


class TempoLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that updates local epochs according to server instructions."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)
        local_epoch_num = server_response.get("local_epoch_num")
        if local_epoch_num is None:
            return

        Config().trainer = Config().trainer._replace(epochs=local_epoch_num)


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a Tempo client using the adaptive lifecycle strategy."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=TempoLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
