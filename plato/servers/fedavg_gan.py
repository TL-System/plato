"""Federated averaging server specialized for GAN training."""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgGanAggregationStrategy


class Server(fedavg.Server):
    """Federated learning server using the GAN-specific aggregation strategy."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
        client_selection_strategy=None,
    ):
        if aggregation_strategy is None:
            aggregation_strategy = FedAvgGanAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )

    def customize_server_payload(self, payload):
        """
        Customize the server payload before sending to the client.

        At the end of each round, the server can choose to only send the global Generator
        or Discriminator (or both or neither) model to the clients next round.

        Reference this paper for more detail:
        https://deepai.org/publication/federated-generative-adversarial-learning

        By default, both model will be sent to the clients.
        """
        if hasattr(Config().server, "network_to_sync"):
            network = Config().server.network_to_sync.lower()
        else:
            network = "both"

        weights_gen, weights_disc = payload
        if network == "none":
            server_payload = None, None
        elif network == "generator":
            server_payload = weights_gen, None
        elif network == "discriminator":
            server_payload = None, weights_disc
        elif network == "both":
            server_payload = payload
        else:
            raise ValueError(f"Unknown value to attribute network_to_sync: {network}")

        return server_payload
