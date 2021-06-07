"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.server_update_direction = None
        self.update_directions_received = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.update_directions_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """

        update = await super().federated_averaging(updates)

        # Initialize server update direction
        if self.server_update_direction is None:
            self.server_update_direction = [0] * len(
                self.update_directions_received[0])

        # Update server update direction
        for update_direction in self.update_directions_received:
            for j, delta in enumerate(update_direction):
                self.server_update_direction[j] += delta / Config(
                ).clients.total_clients

        return update

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.server_update_direction]
