"""
Base class for trainers.
"""

from abc import ABC, abstractmethod
import time
import os
from filelock import FileLock
from config import Config


class Trainer(ABC):
    """Base class for all the trainers."""
    def __init__(self, client_id):
        self.device = Config().device()
        self.client_id = client_id
        """Initialize a global counter of running trainers."""
        if not os.path.exists('./running_trainers'):
            with open('./running_trainers', 'w') as file:
                file.write(str(0))

    def start_training(self):
        """Increment the global counter of running trainers."""

        lock = FileLock('./running_trainers.lock')

        with open('./running_trainers', 'r') as file:
            trainer_count = int(file.read())

        while trainer_count >= Config().trainer.max_concurrency:
            # Wait for a while and check again
            time.sleep(5)
            with open('./running_trainers', 'r') as file:
                trainer_count = int(file.read())

        lock.acquire()
        try:
            open('./running_trainers', 'w').write(str(trainer_count + 1))
        finally:
            lock.release()

    def pause_training(self):
        """Increment the global counter of running trainers."""
        with open('./running_trainers', 'r') as file:
            trainer_count = int(file.read())

        lock = FileLock('./running_trainers' + '.lock')
        lock.acquire()
        try:
            open('./running_trainers', 'w').write(str(trainer_count - 1))
        finally:
            lock.release()

        model_type = Config().trainer.model
        model_dir = Config().model_dir
        model_path = f'{model_dir}{model_type}_{self.client_id}_{Config().experiment_id}.pth'
        accuracy_path = f'{model_dir}{model_type}_{self.client_id}_{Config().experiment_id}.acc'

        if os.path.exists(model_path):
            os.remove(model_path)

        if os.path.exists(accuracy_path):
            os.remove(accuracy_path)

    def stop_training(self):
        """ Remove the global counter after all training concluded."""
        os.remove('./running_trainers')
        os.remove('./running_trainers.lock')

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
    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
