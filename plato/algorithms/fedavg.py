"""
The federated averaging algorithm for PyTorch.
"""
from typing import OrderedDict
import torch
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)

    @staticmethod
    def weights_to_numpy(weights):
        """Converts weights from a model into numpy format."""
        return OrderedDict([(k, v.detach().numpy())
                            for k, v in weights.items()])

    @staticmethod
    def numpy_to_weights(weights):
        """Converts numpy formatted weights into model weight."""
        return OrderedDict([(k, torch.tensor(v)) for k, v in weights.items()])
