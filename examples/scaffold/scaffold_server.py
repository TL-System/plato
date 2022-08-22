"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from collections import OrderedDict

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

    def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregates the model updates using the deltas directly received by SCAFFOLD clients."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

        # Update weights by adding the deltas to the baseline
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + avg_update[name]

        return updated_weights

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated."""
        # Update server control variate
        for client_control_variate_delta in self.received_client_control_variates:
            for name, param in client_control_variate_delta.items():
                self.server_control_variate[name] += param * (
                    1 / Config().clients.total_clients
                )

    def customize_server_payload(self, payload):
        "Add the server control variate into the server payload."
        if self.server_control_variate is None:
            self.server_control_variate = OrderedDict()
            for name, weight in self.algorithm.extract_weights().items():
                self.server_control_variate[name] = self.trainer.zeros(weight.shape)

        return [payload, self.server_control_variate]
