"""
HeteroFL algorithm trainer.
"""

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the Hermes algorithm."""
    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer)
        self.rates = [None for _ in range(Config().clients.total_clients)]

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        server_response["rate"] = self.algorithm.choose_rate()  # need implementation
        return super().customize_server_response(server_response, client_id)

    # implement in server algorithm extract to generate the customized model

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received
    ):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        client_id_list = [update.client_id for update in self.updates]
        self.algorithm.nas_aggregation(
            self.rates, weights_received, client_id_list
        )
        # need implementation
        