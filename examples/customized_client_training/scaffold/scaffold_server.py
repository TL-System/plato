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

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.server_control_variate = None
        self.received_client_control_variates = None

    def weights_received(self, weights_received):
        """Compute control variates from clients' updated weights."""
        # Each weight is [model_weights, Δc_i]. Save Δc_i for Eq. (5) update.
        self.received_client_control_variates = [
            weight[1] for weight in weights_received
        ]
        return [weight[0] for weight in weights_received]

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated.
        Update server control variate per SCAFFOLD Eq. (5):
        c ← c + (1/m) ∑ Δc_i over participating clients.
        """
        deltas = [d for d in self.received_client_control_variates if d is not None]
        if not deltas:
            return
        N = Config().clients.total_clients
        for name in self.server_control_variate:
            incr = sum(d[name].cpu() for d in deltas) * (1.0 / N)
            self.server_control_variate[name] += incr

    def customize_server_payload(self, payload):
        "Add the server control variate into the server payload."
        if self.server_control_variate is None:
            self.server_control_variate = OrderedDict()
            for name, weight in self.algorithm.extract_weights().items():
                self.server_control_variate[name] = self.trainer.zeros(weight.shape)

        return [payload, self.server_control_variate]
