"""
Base class for algorithms.
"""

from abc import ABC, abstractmethod

from plato.trainers.base import Trainer


class Algorithm(ABC):
    """Base class for all the algorithms."""
    def __init__(self, trainer: Trainer, client_id=None):
        """Initializing the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__()
        self.trainer = trainer
        self.model = trainer.model
        self.client_id = client_id

    @abstractmethod
    def extract_weights(self):
        """Extract weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
