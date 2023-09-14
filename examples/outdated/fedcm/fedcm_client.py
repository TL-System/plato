"""
Reference:
J. Xu, et al. "FedCM: Federated Learning with Client-level Momentum," found in papers/.
"""

from plato.config import Config

from plato.clients import simple
from plato.models import registry as models_registry


class Client(simple.Client):
    """A federated learning client with support for Adaptive Synchronization
    Frequency.
    """

    def _load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        self.algorithm.load_weights(server_payload[0])
        if server_payload[1]:
            self.trainer.delta = models_registry.get()
            self.trainer.delta.load_state_dict(server_payload[1], strict=True)
        self.trainer.current_round = server_payload[2]
