"""
A federated learning client at edge server of Tempo.
"""

from plato.clients import edge
from plato.clients.strategies.edge import EdgeLifecycleStrategy
from plato.config import Config


class TempoEdgeLifecycleStrategy(EdgeLifecycleStrategy):
    """Lifecycle strategy that updates edge clients' local epochs."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)

        local_epoch_list = server_response.get("local_epoch_num")
        if local_epoch_list is None:
            return

        logical_client_id = context.client_id
        index = logical_client_id - Config().clients.total_clients - 1

        if isinstance(local_epoch_list, list):
            try:
                local_epoch_num = local_epoch_list[index]
            except (IndexError, TypeError):
                local_epoch_num = (
                    local_epoch_list[0] if local_epoch_list else Config().trainer.epochs
                )
        else:
            local_epoch_num = local_epoch_list

        Config().trainer = Config().trainer._replace(epochs=local_epoch_num)


def create_client(
    *,
    server,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
):
    """Build a Tempo edge client with adaptive lifecycle strategy."""
    client = edge.Client(
        server=server,
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=TempoEdgeLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
