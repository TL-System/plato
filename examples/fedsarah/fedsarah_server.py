"""
A customized server for FedSarah.

Reference: Ngunyen et al., "SARAH: A Novel Method for Machine Learning Problems
Using Stochastic Recursive Gradient." (https://arxiv.org/pdf/1703.00102.pdf)

"""
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedSarah algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.server_control_variates = None
        self.control_variates_received = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and control variates from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.control_variates_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight and delta updates from client updates. """
        update = await super().federated_averaging(updates)

        # Initialize server control variates
        self.server_control_variates = [0] * len(
            self.control_variates_received[0])

        # Update server control variates
        for control_variates in self.control_variates_received:
            for j, control_variate in enumerate(control_variates):
                self.server_control_variates[j] += control_variate / Config(
                ).clients.total_clients

        return update

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.server_control_variates]
