"""
Federared Model Search via Reinforcement Learning

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522
"""
from plato.algorithms import fedavg
from plato.config import Config

from Darts.model_search_local import MaskedNetwork
from fednas_tools import client_weight_param


class FedNASAlgorithm(fedavg.Algorithm):
    """basic algorithm for FedNAS"""

    def generate_client_model(self, mask_normal, mask_reduce):
        """generated the structure of client model"""
        client_model = MaskedNetwork(
            Config().parameters.model.C,
            Config().parameters.model.num_classes,
            Config().parameters.model.layers,
            mask_normal,
            mask_reduce,
        )
        return client_model


class ServerAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedNAS, used by the server, who holds supernet."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal = None
        self.mask_reduce = None

    def extract_weights(self, model=None):
        """Extract weights from the supernet and assign different models to clients."""
        if model is None:
            model = self.model

        mask_normal = self.mask_normal
        mask_reduce = self.mask_reduce
        client_model = self.generate_client_model(mask_normal, mask_reduce)
        client_weight_param(model.model, client_model)
        return client_model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""


class ClientAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedNAS, used by the client, who holds subnets."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal = None
        self.mask_reduce = None
        self.representation_param_names = []

    def extract_weights(self, model=None):
        if model is None:
            model = self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)
