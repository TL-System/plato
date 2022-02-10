"""
The federated averaging algorithm for PyTorch.
"""
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        if model is None:
            return self.model.cpu().state_dict()
        else:
            return model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)
