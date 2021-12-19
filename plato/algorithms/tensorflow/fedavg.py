"""
The federated averaging algorithm for TensorFlow.
"""
from collections import OrderedDict

from plato.algorithms import base
from plato.datasources import registry as datasources_registry
from plato.trainers.base import Trainer


class Algorithm(base.Algorithm):
    """ Framework-specific algorithms for federated Averaging with TensorFlow, used
    by both the client and the server. """
    def __init__(self, trainer: Trainer):
        """Initializing the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__(trainer)
        if hasattr(self.model, 'build_model'):
            self.model.build_model(datasources_registry.get_input_shape())
        else:
            self.model = trainer.model

    def extract_weights(self):
        """ Extract weights from the model. """
        return self.model.get_weights()

    def compute_weight_updates(self, weights_received):
        """ Extract the weights received from a client and compute the updates. """
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for index, current_weight in enumerate(weight):
                baseline = baseline_weights[index]

                # Calculate update
                delta = current_weight - baseline
                update[index] = delta
            updates.append(update)

        return updates

    def update_weights(self, update):
        """ Update the existing model weights. """
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for index, weight in enumerate(baseline_weights):
            updated_weights[index] = weight + update[index]

        return updated_weights

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.set_weights(weights)
