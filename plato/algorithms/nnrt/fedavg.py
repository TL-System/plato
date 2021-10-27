"""
The federated averaging algorithm for NNRT.
"""
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """NNRT-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)
