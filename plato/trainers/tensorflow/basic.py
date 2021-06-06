"""
The training and testing loop.
"""

import logging
import multiprocessing as mp
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import wandb

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
        model_dir = Config().params['model_dir']

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
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        if 'use_wandb' in config:
            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        custom_train = getattr(self, "train_model", None)

        if callable(custom_train):
            self.train_model(config, trainset, sampler.get(), cut_layer)
        else:
            # Initializing the loss criterion
            _loss_criterion = getattr(self, "loss_criterion", None)
            if callable(_loss_criterion):
                loss_criterion = self.loss_criterion(self.model)
            else:
                loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)

            # Initializing the optimizer
            get_optimizer = getattr(self, "get_optimizer", None)
            if callable(get_optimizer):
                optimizer = self.get_optimizer(self.model)
            else:
                optimizer = tf.keras.optimizers.Adam(config['learning_rate'])

            self.model.compile(
                optimizer=optimizer,
                loss=loss_criterion,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            self.model.fit(trainset, epochs=config['epochs'])

        model_type = Config().trainer.model_name
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.ckpt"
        self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()

    def test_process(self, config, testset):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        rank: Required by torch.multiprocessing to spawn processes. Unused.
        testset: The test dataset.
        """
        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                accuracy = self.model.evaluate(testset, verbose=0)[1]
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        model_name = config['model_name']
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
        Trainer.save_accuracy(accuracy, filename)

    def train(self, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        """
        self.start_training()

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        proc = mp.Process(target=Trainer.train_process,
                          args=(
                              self,
                              config,
                              trainset,
                              sampler,
                              cut_layer,
                          ))
        proc.start()
        proc.join()

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.ckpt"
        self.load_model(filename)

        self.pause_training()

    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        self.start_training()

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        proc = mp.Process(target=Trainer.test_process,
                          args=(
                              self,
                              config,
                              testset,
                          ))
        proc.start()
        proc.join()

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
        accuracy = Trainer.load_accuracy(filename)

        self.pause_training()
        return accuracy
