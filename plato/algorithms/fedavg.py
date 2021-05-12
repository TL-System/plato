"""
The federated averaging algorithm for PyTorch.
"""
from collections import OrderedDict

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.state_dict()

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                delta = current_weight - baseline
                update[name] = delta
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)
