"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from collections import OrderedDict
import torch

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.server_control_variate = None
        self.received_client_control_variates = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and control variates from clients' updates. """
        weights_received = [payload[0] for (__, payload, __) in updates]

        self.received_client_control_variates = [
            payload[1] for (__, payload, __) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and also update server control variate. """

        update = await super().federated_averaging(updates)

        # Update server control variate
        for client_control_variate_delta in self.received_client_control_variates:
            for name, param in client_control_variate_delta.items():
                self.server_control_variate[name].add_(
                    param, alpha=1 / Config().clients.total_clients)

        return update

    def customize_server_payload(self, payload):
        "Add the server control variate into the server payload."
        if self.server_control_variate is None:
            self.server_control_variate = OrderedDict()
            for name, weight in self.algorithm.extract_weights().items():
                self.server_control_variate[name] = torch.zeros(weight.shape)

        return [payload, self.server_control_variate]
