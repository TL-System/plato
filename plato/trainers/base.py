"""
Base class for trainers.
"""

from abc import ABC, abstractmethod
import os

from plato.config import Config


class Trainer(ABC):
    """Base class for all the trainers."""

    def __init__(self):
        self.device = Config().device()
        self.client_id = 0

    def set_client_id(self, client_id):
        """Setting the client ID."""
        self.client_id = client_id

    @abstractmethod
    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        raise TypeError("save_model() not implemented.")

    @abstractmethod
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        raise TypeError("load_model() not implemented.")

    @staticmethod
    def save_accuracy(accuracy, filename=None):
        """Saving the test accuracy to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.acc"

        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(accuracy))

    @staticmethod
    def load_accuracy(filename=None):
        """Loading the test accuracy from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.acc"

        with open(accuracy_path, "r", encoding="utf-8") as file:
            accuracy = float(file.read())

        return accuracy

    def pause_training(self):
        """Remove files of running trainers."""
        if hasattr(Config().trainer, "max_concurrency"):
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            model_file = f"{model_path}/{model_name}_{self.client_id}_{Config().params['run_id']}.pth"
            accuracy_file = f"{model_path}/{model_name}_{self.client_id}_{Config().params['run_id']}.acc"

            if os.path.exists(model_file):
                os.remove(model_file)
                os.remove(model_file + ".pkl")

            if os.path.exists(accuracy_file):
                os.remove(accuracy_file)

    @abstractmethod
    def train(self, trainset, sampler, **kwargs) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.

        Returns:
        float: The training time.
        """

    @abstractmethod
    def test(self, testset, sampler=None, **kwargs) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
