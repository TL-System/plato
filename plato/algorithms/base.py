"""
Base class for algorithms.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

from plato.trainers.base import Trainer


class Algorithm(ABC):
    """Base class for all the algorithms."""
    def __init__(self, trainer: Trainer):
        """Initializing the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__()
        self.trainer = trainer
        self.model = trainer.model
        self.client_id = 0

    def set_client_id(self, client_id):
        """ Setting the client ID. """
        self.client_id = client_id

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

    def update_weights(self, update):
        """ Update the existing model weights. """
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + update[name]

        return updated_weights

    @abstractmethod
    def extract_weights(self):
        """Extract weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
