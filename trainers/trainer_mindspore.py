"""
The training and testing loop.
"""

import logging
import os
import mindspore.nn as nn
import mindspore
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
        """
        super().__init__(client_id)
        self.model = model
        self.mindspore_model = None

        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE,
                                      device_target='GPU')

    def save_model(self):
        """Saving the model to a file."""
        model_type = Config().trainer.model
        model_dir = './models/pretrained/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{model_dir}{model_type}_{self.client_id}.ckpt'
        mindspore.save_checkpoint(self.model, model_path)

        logging.info('[Client #%s] Model saved to %s.', self.client_id,
                     model_path)

    def load_model(self, model_type):
        """Loading pre-trained model weights from a file."""
        model_dir = './models/pretrained/'
        model_path = f'{model_dir}{model_type}_{self.client_id}.ckpt'
        logging.info("[Client #%s] Loading model from %s.", self.client_id,
                     model_path)
        param_dict = mindspore.load_checkpoint(model_path)
        mindspore.load_param_into_net(self.model, param_dict)

    def extract_weights(self):
        """Extract weights from the model."""
        weights = {}
        for _, param in self.model.parameters_and_names():
            weights[param.name] = param

        return weights

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = []
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                delta = current_weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        updated_state_dict = {}
        for name, weight in weights.items():
            updated_state_dict[name] = weight

        mindspore.load_param_into_net(self.model, updated_state_dict)

    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        self.start_training()

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

        self.mindspore_model.train(Config().trainer.epochs,
                                   trainset,
                                   callbacks=[LossMonitor()])

        self.pause_training()

    def test(self, testset, cut_layer=None):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
        self.start_training()

        accuracy = self.mindspore_model.eval(testset, dataset_sink_mode=False)

        self.pause_training()
        return accuracy['Accuracy']
