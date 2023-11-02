"""
The federated averaging algorithm for PyTorch.
"""

from collections import OrderedDict

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""

    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Compute the deltas between baseline weights and weights received."""
        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                _delta = current_weight - baseline
                delta[name] = _delta
            deltas.append(delta)

        return deltas

    def update_weights(self, deltas):
        """Updates the existing model weights from the provided deltas."""
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + deltas[name]

        return updated_weights

    def extract_weights(self, model=None):
        """Extracts weights from the model."""
        if model is None:
            return self.model.cpu().state_dict()
        else:
            return model.cpu().state_dict()

    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)
