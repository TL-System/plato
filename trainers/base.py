"""
Base class for trainers.
"""

from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def extract_weights(self):
        """Extract weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""

    @abstractmethod
    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

    @abstractmethod
    def test(self, testset, batch_size, cut_layer=None):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        batch_size: the batch size used for testing.
        cut_layer (optional): The layer which testing should start from.
        """