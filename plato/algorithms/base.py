"""
Base class for algorithms.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from plato.trainers.base import Trainer


class Algorithm(ABC):
    """Base class for all the algorithms."""

    def __init__(self, trainer: Optional[Trainer]):
        """Initializes the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__()
        self.trainer: Optional[Trainer] = trainer
        self.model: Optional[Any] = getattr(trainer, "model", None) if trainer else None
        self.client_id = 0

    def __repr__(self):
        if self.client_id == 0:
            return f"Server #{os.getpid()}"
        else:
            return f"Client #{self.client_id}"

    def set_client_id(self, client_id):
        """Sets the client ID."""
        self.client_id = client_id

    @abstractmethod
    def extract_weights(self, model=None):
        """Extracts weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""

    async def aggregate_weights(self, baseline_weights, weights_received, **kwargs):
        """Aggregates the weights received into baseline weights (optional)."""
