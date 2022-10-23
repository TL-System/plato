from plato.algorithms import fedavg
from plato.config import Config

from Darts.model_search_local import MaskedNetwork
from fednas_tools import *


class ServerAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        # parameter names of the representation
        # As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        # behaves as the global model.
        self.mask_normal = None
        self.mask_reduce = None

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        if model is None:
            model = self.model

        mask_normal = self.mask_normal
        mask_reduce = self.mask_reduce
        client_model = MaskedNetwork(
            Config().parameters.model.C,
            Config().parameters.model.num_classes,
            Config().parameters.model.layers,
            mask_normal,
            mask_reduce,
        )
        client_weight_param(model.model, client_model)
        return client_model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""


class ClientAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

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
