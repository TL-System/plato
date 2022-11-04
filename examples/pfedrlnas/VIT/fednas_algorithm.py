from collections import OrderedDict

from plato.algorithms import fedavg
import fedtools
from plato.config import Config
import pickle


class ServerAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_subnet = None

    def extract_weights(self, model=None):
        payload = self.current_subnet.cpu().state_dict()
        return payload

    def load_weights(self, weights):
        pass


class ClientAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

    def extract_weights(self, model=None):
        if model is None:
            model = self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)
