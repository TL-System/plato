"""
The training and testing loop.
"""

import logging
import os
from collections import OrderedDict
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from models.base_mindspore import Model
from config import Config
from trainers import base


class Trainer(base.Trainer):
    """A basic federated learning trainer for MindSpore, used by both
    the client and the server.
    """
    def __init__(self, model: Model, client_id=0):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train. Must be a models.base_mindspore.Model subclass.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(client_id)

        mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,
                                      device_target='GPU')

        self.model = model

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

    def save_model(self):
        """Saving the model to a file."""
        model_type = Config().trainer.model
        model_dir = './models/pretrained/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{model_dir}{model_type}_{self.client_id}.ckpt'
        mindspore.save_checkpoint(self.model, model_path)

        if self.client_id == 0:
            logging.info("[Server #%s] Model saved to %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%s] Model saved to %s.", self.client_id,
                         model_path)

    def load_model(self):
        """Loading pre-trained model weights from a file."""
        model_dir = './models/pretrained/'
        model_type = Config().trainer.model
        model_path = f'{model_dir}{model_type}_{self.client_id}.ckpt'

        if self.client_id == 0:
            logging.info("[Server #%s] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%s] Loading a model from %s.",
                         self.client_id, model_path)

        param_dict = mindspore.load_checkpoint(model_path)
        mindspore.load_param_into_net(self.model, param_dict)

    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.parameters_dict()

    def print_weights(self):
        """Print all the weights from the model."""
        for _, param in self.model.parameters_and_names():
            print(f'key = {param.name}, value = {param.asnumpy()}')

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                delta = current_weight - baseline
                update[name] = delta
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name, weight in weights.items():
            weights[name] = mindspore.Parameter(weight, name=name)

        # One can also use `self.model.load_parameter_slice(weights)', which
        # seems to be equivalent to mindspore.load_param_into_net() in its effects

        mindspore.load_param_into_net(self.model, weights, strict_load=True)

    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        self.start_training()

        self.mindspore_model.train(
            Config().trainer.epochs,
            trainset,
            callbacks=[LossMonitor(per_print_times=300)])

        self.pause_training()

    def test(self, testset, cut_layer=None):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
        self.start_training()

        accuracy = self.mindspore_model.eval(testset)

        self.pause_training()
        return accuracy['Accuracy']
