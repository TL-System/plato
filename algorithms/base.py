"""
Base class for algorithms.
"""

from abc import ABC, abstractmethod

from models.base import Model
from trainers.base import Trainer


class Algorithm(ABC):
    """Base class for all the algorithms."""
    def __init__(self, model: Model, trainer: Trainer = None, client_id=None):
        """Initializing the algorithm with the provided model and trainer.

        Arguments:
        model: The model to train, which is a models.base.Model class.
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        """
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.client_id = client_id

    @abstractmethod
    def extract_weights(self):
        """Extract weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
