"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, List, Optional

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
        self.server_control_variate: Optional[OrderedDict[str, Any]] = None
        self.received_client_control_variates: Optional[
            List[Optional[OrderedDict[str, Any]]]
        ] = None

    def weights_received(self, weights_received: List[Sequence[Any]]) -> List[Any]:
        """Compute control variates from clients' updated weights."""
        # Each weight is [model_weights, Δc_i]. Save Δc_i for Eq. (5) update.
        self.received_client_control_variates = [
            weight[1] if len(weight) > 1 else None for weight in weights_received
        ]
        return [weight[0] for weight in weights_received]

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated.
        Update server control variate per SCAFFOLD Eq. (5):
        c ← c + (1/m) ∑ Δc_i over participating clients.
        """
        variates = self.received_client_control_variates
        if not variates:
            return

        deltas = [d for d in variates if d is not None]
        if not deltas:
            return

        server_control_variate = self.server_control_variate
        if server_control_variate is None:
            raise RuntimeError(
                "Server control variate must be initialized before aggregation."
            )

        N = Config().clients.total_clients
        for name in server_control_variate:
            incr = sum(d[name].cpu() for d in deltas) * (1.0 / N)
            server_control_variate[name] += incr

    def customize_server_payload(self, payload):
        "Add the server control variate into the server payload."
        server_control_variate = self.server_control_variate

        if server_control_variate is None:
            algorithm = self.algorithm
            if algorithm is None or not hasattr(algorithm, "extract_weights"):
                raise RuntimeError(
                    "SCAFFOLD requires an algorithm with an extract_weights method."
                )
            weights = algorithm.extract_weights()
            trainer = self.trainer
            if trainer is None or not hasattr(trainer, "zeros"):
                raise RuntimeError(
                    "SCAFFOLD requires a trainer that provides a zeros factory method."
                )

            server_control_variate = OrderedDict()
            for name, weight in weights.items():
                server_control_variate[name] = trainer.zeros(weight.shape)
            self.server_control_variate = server_control_variate

        return [payload, server_control_variate]
