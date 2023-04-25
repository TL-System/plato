"""
The training and testing loop.
"""

import logging
import os
import time

import tensorflow as tf

from plato.config import Config
from plato.trainers import base
from plato.models import registry as models_registry


class Trainer(base.Trainer):
    """A basic federated learning trainer for TensorFlow, used by both
    the client and the server.
    """

    def __init__(self, model=None, **kwargs):
        """Initializing the trainer with the provided model.

        Arguments:
        client_id: The ID of the client using this trainer (optional).
        model: The model to train.
        """
        super().__init__()

        if model is None:
            self.model = models_registry.get()
        else:
            self.model = model()

    def zeros(self, shape):
        """Returns a TensorFlow zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return tf.zeros(shape)

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.ckpt"

        self.model.save_weights(model_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id, model_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.ckpt"

        if self.client_id == 0:
            logging.info(
                "[Server #%d] Loading a model from %s.", os.getpid(), model_path
            )
        else:
            logging.info(
                "[Client #%d] Loading a model from %s.", self.client_id, model_path
            )

        self.model.load_weights(model_path)

    def train_process(self, config, trainset, sampler):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        """
        try:
            self.train_model(config, trainset, sampler.get())
        except Exception as training_exception:
            logging.info(
                "Training on client #%d failed: %s", self.client_id, training_exception
            )
            raise training_exception

    def train(self, trainset, sampler, **kwargs) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]
        if hasattr(Config().trainer, "max_concurrency"):
            # reserved for mp.Process
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler)
            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """Trains the model."""
        # Initializing the loss criterion
        loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer", None)
        if callable(get_optimizer):
            optimizer = self.get_optimizer(self.model)
        else:
            optimizer = tf.keras.optimizers.Adam(Config().parameters.optimizer.lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_criterion,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        logging.info("Begining training on client #%d.", self.client_id)
        self.model.fit(trainset, epochs=config["epochs"])

    def test(self, testset, sampler=None, **kwargs):
        """Tests the model on the client using the provided test dataset.

        :param testset: the test dataset.
        :param sampler: the test dataset sampler.
        """

        config = Config().trainer._asdict()

        try:
            accuracy = self.test_model(config, testset, sampler)
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        return accuracy

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """Tests the model. Must be compiled first."""
        logging.info("Get loss_criterion on client #%d.", self.client_id)
        loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        logging.info("Get_optimizer on client #%d.", self.client_id)
        get_optimizer = getattr(self, "get_optimizer", None)
        if callable(get_optimizer):
            optimizer = self.get_optimizer(self.model)
        else:
            optimizer = tf.keras.optimizers.Adam(Config().parameters.optimizer.lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_criterion,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        return self.model.evaluate(testset, verbose=0)[1]

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return tf.keras.losses.SparseCategoricalCrossentropy()
