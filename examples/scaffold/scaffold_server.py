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

    def weights_received(self, weights_received):
        """Compute control variates from clients' updated weights."""
        self.received_client_control_variates = [
            weight[1] for weight in weights_received
        ]

        return [weight[0] for weight in weights_received]

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated."""
        # Update server control variate
        for client_control_variate_delta in self.received_client_control_variates:
            for name, param in client_control_variate_delta.items():
                self.server_control_variate[name].add_(
                    param, alpha=1 / Config().clients.total_clients
                )

    def customize_server_payload(self, payload):
        "Add the server control variate into the server payload."
        if self.server_control_variate is None:
            self.server_control_variate = OrderedDict()
            for name, weight in self.algorithm.extract_weights().items():
                self.server_control_variate[name] = torch.zeros(weight.shape)

        return [payload, self.server_control_variate]
