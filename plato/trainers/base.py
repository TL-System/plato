"""
Base class for trainers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from plato.config import Config
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path


class Trainer(ABC):
    """Base class for all the trainers."""

    def __init__(self):
        self.device = Config().device()
        self.client_id = 0
        self.current_round = 0
        # Subclasses populate the actual model instance during initialization.
        self.model: Any | None = None

    def __getattr__(self, name: str) -> Any:
        """Allow dynamic trainer attributes for runtime-extended implementations."""
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}.")

    def set_client_id(self, client_id):
        """Setting the client ID."""
        self.client_id = client_id

    def require_model(self) -> Any:
        """Return the model instance, ensuring it is available."""
        if self.model is None:
            raise RuntimeError("Trainer model has not been initialised.")

        return self.model

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

        accuracy_path = checkpoint_path(
            model_path, filename if filename is not None
            else checkpoint_name(model_name, suffix=".acc"),
        )

        os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(accuracy))

    @staticmethod
    def load_accuracy(filename=None):
        """Loading the test accuracy from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        accuracy_path = checkpoint_path(
            model_path, filename if filename is not None
            else checkpoint_name(model_name, suffix=".acc"),
        )

        with open(accuracy_path, encoding="utf-8") as file:
            accuracy = float(file.read())

        return accuracy

    def pause_training(self):
        """Remove files of running trainers."""
        if hasattr(Config().trainer, "max_concurrency"):
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            components = (model_name, self.client_id, Config().params["run_id"])
            primary = checkpoint_name(*components, suffix=".safetensors")
            filenames = [primary, primary + ".pkl"]
            filenames.extend(
                checkpoint_name(*components, suffix=suffix)
                for suffix in (".acc", ".eval.pkl", ".train.pkl")
            )
            for filename in filenames:
                path = checkpoint_path(model_path, filename)
                if os.path.exists(path):
                    os.remove(path)

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
