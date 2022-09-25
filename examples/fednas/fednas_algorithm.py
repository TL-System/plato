"""
A personalized federated learning training algorithm using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

from collections import OrderedDict

from plato.algorithms import fedavg
from fednas_tools import*
from Darts.model_search_local import MaskedNetwork
from plato.config import Config
import pickle

class ServerAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        # parameter names of the representation
        # As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        # behaves as the global model.
        self.mask_normal=None
        self.mask_reduce=None

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        if model is None:
            model=self.model

        mask_normal = self.mask_normal
        mask_reduce = self.mask_reduce
        client_model= MaskedNetwork(Config().parameters.model.C,Config().parameters.model.num_classes,Config().parameters.model.layers,nn.CrossEntropyLoss(),mask_normal,mask_reduce)
        client_weight_param(model.model,client_model)
        return client_model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.model.load_state_dict(weights, strict=True)

class ClientAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedRep, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal=None
        self.mask_reduce=None
        self.representation_param_names = []

    def extract_weights(self, model=None):
        if model is None:
            model=self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights,strict=True)