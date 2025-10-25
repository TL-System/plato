"""
Base class for algorithms.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from plato.trainers.base import Trainer


class Algorithm(ABC):
    """Base class for all the algorithms."""

    def __init__(self, trainer: Trainer | None):
        """Initializes the algorithm with the provided model and trainer.

        Arguments:
        trainer: The trainer for the model, which is a trainers.base.Trainer class.
        model: The model to train.
        """
        super().__init__()
        self.trainer: Trainer | None = trainer
        self.model: Any | None = getattr(trainer, "model", None) if trainer else None
        self.client_id = 0

    def __repr__(self):
        if self.client_id == 0:
            return f"Server #{os.getpid()}"
        else:
            return f"Client #{self.client_id}"

    def __getattr__(self, name: str) -> Any:
        """Permit dynamic attributes injected by specific algorithms."""
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}.")

    def set_client_id(self, client_id: int) -> None:
        """Sets the client ID."""
        self.client_id = client_id

    def require_trainer(self) -> Trainer:
        """Return the trainer instance, ensuring it is available."""
        if self.trainer is None:
            raise RuntimeError(
                "Trainer is not attached to the algorithm; cannot continue."
            )
        return self.trainer

    def require_model(self) -> Any:
        """Return the model instance, ensuring it is available."""
        if self.model is None:
            raise RuntimeError(
                "Model is not attached to the algorithm; cannot continue."
            )
        return self.model

    @abstractmethod
    def extract_weights(self, model=None):
        """Extracts weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""

    async def aggregate_weights(self, baseline_weights, weights_received, **kwargs):
        """Aggregates the weights received into baseline weights (optional)."""
