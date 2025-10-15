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


class Client(edge.Client):
    """A federated learning edge client of Tempo."""

    def __init__(
        self,
        server,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            server=server,
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=TempoEdgeLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )
