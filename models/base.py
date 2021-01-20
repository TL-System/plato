"""Base classes for the model."""

from abc import ABC, abstractmethod, abstractstaticmethod
import torch.nn as nn


class Model(ABC, nn.Module):
    """The base class for by all the models."""
    @abstractmethod
    def forward(self, x):
        """The forward pass."""

    @abstractstaticmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

    @abstractstaticmethod
    def get_model_from_name(model_name: str) -> 'Model':
        """Returns an instance of this class as described by the model_name string."""
