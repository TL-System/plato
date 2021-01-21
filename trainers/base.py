"""
Base class for trainers.
"""

from abc import ABC, abstractmethod
import time
import os
from config import Config


class Trainer(ABC):
    """Base class for all the trainers."""
    def __init__(self):
        self.device = Config().device()
        """Initialize a global counter of running trainers."""
        if not os.path.exists('./running_trainers'):
            with open('./running_trainers', 'w') as file:
                file.write(str(0))

    def started_training(self):
        """Increment the global counter of running trainers."""
        with open('./running_trainers', 'r') as file:
            trainer_count = int(file.read())

        while trainer_count >= Config().trainer.max_concurrency:
            # Wait for a while and check again
            time.sleep(5)
            with open('./running_trainers', 'r') as file:
                trainer_count = int(file.read())

        with open('./running_trainers', 'w') as file:
            file.write(str(trainer_count + 1))

    def paused_training(self):
        """Increment the global counter of running trainers."""
        with open('./running_trainers', 'r') as file:
            trainer_count = int(file.read())
        with open('./running_trainers', 'w') as file:
            file.write(str(trainer_count - 1))

    def stopped_training(self):
        os.remove('./running_trainers')

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
