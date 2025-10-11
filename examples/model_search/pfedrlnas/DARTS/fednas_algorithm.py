"""
Implement new algorithm: personalized federarted NAS.
"""

from Darts.model_search_local import MaskedNetwork
from fednas_tools import (
    client_weight_param,
    extract_index,
    fuse_weight_gradient,
    sample_mask,
)

from plato.algorithms import fedavg
from plato.config import Config


class FedNASAlgorithm(fedavg.Algorithm):
    """Basic algorithm for FedRLNAS."""

    def generate_client_model(self, mask_normal, mask_reduce):
        """Generates the structure of the client model."""
        client_model = MaskedNetwork(
            Config().parameters.model.C,
            Config().parameters.model.num_classes,
            Config().parameters.model.layers,
            mask_normal,
            mask_reduce,
        )

        return client_model


class ServerAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedRLNAS, used by the server."""

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

    def sample_mask(self, client_id):
        """Sample mask to generate a subnet."""
        client_id -= 1
        mask_normal = sample_mask(self.model.alphas_normal[client_id])
        mask_reduce = sample_mask(self.model.alphas_reduce[client_id])
        self.mask_normal = mask_normal
        self.mask_reduce = mask_reduce
        return mask_normal, mask_reduce

    def nas_aggregation(
        self, masks_normal, masks_reduce, weights_received, num_samples
    ):
        """Weight aggregation in NAS."""
        client_models = []

        for i, payload in enumerate(weights_received):
            mask_normal = masks_normal[i]
            mask_reduce = masks_reduce[i]
            client_model = self.generate_client_model(mask_normal, mask_reduce)
            client_model.load_state_dict(payload, strict=True)
            client_models.append(client_model)
        fuse_weight_gradient(
            self.model.model,
            client_models,
            num_samples,
        )

    def extract_index(self, mask):
        """Extract edge index according to the mask."""
        return extract_index(mask)


class ClientAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedRLNAS, used by the client."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal = None
        self.mask_reduce = None

    def extract_weights(self, model=None):
        if model is None:
            model = self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)
