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
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        client_id: The ID of the client using this trainer (optional).
        model: The model to train.
        """
        super().__init__()

        if model is None:
            self.model = models_registry.get()
        else:
            self.model = model

    def zeros(self, shape):
        """Returns a TensorFlow zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return tf.zeros(shape)

    def save_model(self, filename=None):
        """Saving the model to a file."""
        model_name = Config().trainer.model_name
        model_dir = Config().params['model_dir']

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.ckpt'

        self.model.save_weights(model_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id,
                         model_path)

    def load_model(self, filename=None):
        """Loading pre-trained model weights from a file."""
        model_name = Config().trainer.model_name
        model_dir = Config().params['pretrained_model_dir']

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.ckpt'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s.",
                         self.client_id, model_path)

        self.model.load_weights(model_path)

    def train_process(self, config, trainset, sampler, cut_layer=None):
        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        custom_train = getattr(self, "train_model", None)
        try:
            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                if 'is_compiled' not in config:
                    # Initializing the loss criterion
                    logging.info("Get loss_criterion on client #%d.",
                                 self.client_id)
                    _loss_criterion = getattr(self, "loss_criterion", None)
                    if callable(_loss_criterion):
                        loss_criterion = self.loss_criterion(self.model)
                    else:
                        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
                        )

                    # Initializing the optimizer
                    logging.info("Get_optimizer on client #%d.",
                                 self.client_id)
                    get_optimizer = getattr(self, "get_optimizer", None)
                    if callable(get_optimizer):
                        optimizer = self.get_optimizer(self.model)
                    else:
                        optimizer = tf.keras.optimizers.Adam(
                            config['learning_rate'])

                    self.model.compile(
                        optimizer=optimizer,
                        loss=loss_criterion,
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

                logging.info("Begining training on client #%d.",
                             self.client_id)
                self.model.fit(trainset, epochs=config['epochs'])
        except Exception as training_exception:
            logging.info("Training on client #%d failed: %s", self.client_id,
                         training_exception)
            raise training_exception

        if 'use_wandb' in config:
            run.finish()

    def train(self, trainset, sampler, cut_layer=None) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']
        if hasattr(Config().trainer, 'max_concurrency'):
            # reserved for mp.Process
            self.start_training()
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, cut_layer)
            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, cut_layer)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    def test(self, testset, sampler=None):
        """Testing the model on the client using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        custom_test = getattr(self, "test_model", None)

        if callable(custom_test):
            config = Config().trainer._asdict()
            accuracy = self.test_model(config, testset)
        else:
            accuracy = self.model.evaluate(testset, verbose=0)[1]

        return accuracy

    async def server_test(self, testset):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        if not hasattr(Config().trainer, 'is_compiled'):
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    Config().trainer.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        return self.test(testset)
