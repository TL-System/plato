"""Base classes for PyTorch models."""

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

    def is_train_process(self):
        return False
        """Does the model has a train_process?"""

    def is_test_process(self):
        return False
        """Does the model has a test_process?"""

    # @abstractstaticmethod
    def train_process(rank, self, config, trainset, cut_layer=None):
        "Train_process"

    def test_process(rank, self, config, testset):
        "Test_process"

