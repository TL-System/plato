"""
The federated averaging algorithm for TensorFlow.
"""
from collections import OrderedDict

from plato.algorithms import base
from plato.datasources import registry as datasources_registry
from plato.trainers.base import Trainer


class Algorithm(base.Algorithm):
    """Framework-specific algorithms for federated Averaging with TensorFlow, used
    by both the client and the server."""

    def __init__(self, trainer: Trainer):
        """Initializing the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__(trainer)
        if hasattr(self.model, "build_model"):
            self.model.build_model(datasources_registry.get_input_shape())
        else:
            self.model = trainer.model

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        if model is None:
            return self.model.get_weights()

        return model.get_weights()

    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Compute the deltas between baseline weights and weights received."""
        # Calculate deltas from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for index, current_weight in enumerate(weight):
                baseline = baseline_weights[index]

                # Calculating the delta
                _delta = current_weight - baseline
                delta[index] = _delta
            deltas.append(delta)

        return deltas

    def update_weights(self, deltas):
        """Update the existing model weights."""
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for index, weight in enumerate(baseline_weights):
            updated_weights[index] = weight + deltas[index]

        return updated_weights

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.set_weights(weights)
