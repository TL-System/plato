"""
The training and testing loop.
"""

import logging
import os
import time

import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import base

class Trainer(base.Trainer):
    """A basic federated learning trainer for MindSpore, used by both
    the client and the server.
    """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        client_id: The ID of the client using this trainer (optional).
        model: The model to train.
        """
        super().__init__()

        if hasattr(Config().trainer, 'cpuonly') and Config().trainer.cpuonly:
            mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,
                                        device_target='CPU')
        else:
            mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,
                                        device_target='GPU')

        if model is None:
            self.model = models_registry.get()

        # Initializing the loss criterion
        loss_criterion = SoftmaxCrossEntropyWithLogits(sparse=True,
                                                       reduction='mean')

        # Initializing the optimizer
        optimizer = nn.Momentum(self.model.trainable_params(),
                                Config().trainer.learning_rate,
                                Config().trainer.momentum)

        self.mindspore_model = mindspore.Model(
            self.model,
            loss_criterion,
            optimizer,
            metrics={"Accuracy": Accuracy()})

    def zeros(self, shape):
        """Returns a MindSpore zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return mindspore.Tensor(np.zeros(shape), mindspore.float32)

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

        mindspore.save_checkpoint(self.model, model_path)

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

        param_dict = mindspore.load_checkpoint(model_path)
        mindspore.load_param_into_net(self.model, param_dict)

    def train(self, trainset, *args)  -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        """
        self.start_training()
        tic = time.perf_counter()

        self.mindspore_model.train(
            Config().trainer.epochs,
            trainset,
            callbacks=[LossMonitor(per_print_times=300)],
            dataset_sink_mode=False)

        toc = time.perf_counter()
        self.pause_training()
        training_time = toc - tic

        return training_time

    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        self.start_training()

        # Deactivate the cut layer so that testing uses all the layers
        self.mindspore_model._network.cut_layer = None

        accuracy = self.mindspore_model.eval(testset)

        self.pause_training()
        return accuracy['Accuracy']

    async def server_test(self, testset):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        return self.test(testset)
