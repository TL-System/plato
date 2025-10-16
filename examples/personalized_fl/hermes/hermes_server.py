"""
A federated learning server using Hermes.
"""

import hermes_pruning as pruning

from plato.servers import fedavg_personalized as personalized_server
from plato.servers.strategies.aggregation import HermesAggregationStrategy


class Server(personalized_server.Server):
    """A federated learning server using the Hermes algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
    ):
        if aggregation_strategy is None:
            aggregation_strategy = HermesAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
        )
        self.masks_received = []
        self.aggregated_clients_model = {}
        self.total_samples = 0

    def update_client_model(self, aggregated_clients_models, updates):
        """Update clients' models."""
        for client_model, update in zip(aggregated_clients_models, updates):
            received_client_id = update.client_id
            if received_client_id in self.aggregated_clients_model:
                self.aggregated_clients_model[received_client_id] = client_model

    def customize_server_payload(self, payload):
        """Customizes the server payload before sending to the client."""

        # If the client has already begun training a personalized model
        # in a previous communication round, the personalized file is loaded and
        # sent to the client for continued training. Otherwise, if the client is
        # selected for the first time, it receives the pre-initialized model.
        if self.selected_client_id in self.aggregated_clients_model:
            # replace the payload for the current client with the personalized model
            payload = self.aggregated_clients_model[self.selected_client_id]

        return payload

    def weights_received(self, weights_received):
        """Event called after the updated weights have been received."""
        # Extract the model weight updates from client updates along with the masks
        self.masks_received = [payload[1] for payload in weights_received]
        weights = [payload[0] for payload in weights_received]
        for step, mask in enumerate(self.masks_received):
            if mask is None:
                mask = pruning.make_init_mask(self.trainer.model)
                self.masks_received[step] = mask

        return weights
