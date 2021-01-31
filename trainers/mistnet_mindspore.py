"""
The federated learning trainer for MistNet, used by both the client and the
server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import time
import logging
from mindspore.nn import optim
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from models.base_mindspore import Model

from utils import unary_encoding
from trainers import trainer_mindspore
import models.registry as models_registry
from config import Config

class Trainer(trainer_mindspore.Trainer):
    """A federated learning trainer for MistNet in the MindSpore framework, used
    by both the client and the server.
    """
    def __init__(self, model: Model, client_id=0):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train. Must be a models.base_mindspore.Model subclass.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model, client_id)

        # Obtain the full model for testing
        model_type = Config().trainer.model
        self.test_model = models_registry.get(model_type, cut_layer=None)

        # Initializing the loss criterion
        loss_criterion = SoftmaxCrossEntropyWithLogits(sparse=True,
                                                       reduction='mean')

        # Initializing the optimizer
        optimizer = nn.Momentum(self.test_model.trainable_params(),
                                Config().trainer.learning_rate,
                                Config().trainer.momentum)

        self.mindspore_test_model = mindspore.Model(
            self.test_model,
            loss_criterion,
            optimizer,
            metrics={"Accuracy": Accuracy()})

    def extract_features(self, dataset, cut_layer, epsilon=None):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        epsilon: If epsilon is not None, local differential privacy should be
                applied to the features extracted.
        """
        self.model.set_train(False)

        tic = time.perf_counter()

        feature_dataset = []

        for inputs, targets in dataset:
            inputs = mindspore.Tensor(inputs)
            targets = mindspore.Tensor(targets)

            logits = self.model.forward_to(inputs, cut_layer)

            if epsilon is not None:
                logits = logits.asnumpy()
                logits = unary_encoding.encode(logits)
                logits = unary_encoding.randomize(logits, epsilon)
                logits = mindspore.Tensor(logits.astype('float32'))

            for i in np.arange(logits.shape[0]): # each sample in the batch
                logit = mindspore.Tensor(logits.asnumpy()[i])
                target = mindspore.Tensor(targets.asnumpy()[i])
                feature_dataset.append((logit, target))

        toc = time.perf_counter()
        logging.info("[Client #%s] Features extracted from %s examples.",
            self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    @staticmethod
    def dataset_generator(feature_dataset):
        for logit, target in feature_dataset:
            yield (logit.asnumpy(), target.asnumpy())

    def train(self, trainset, cut_layer=None):
        feature_dataset = ds.GeneratorDataset(
            Trainer.dataset_generator(trainset), ["logit", "target"])

        super().train(feature_dataset, cut_layer)

    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        self.start_training()

        accuracy = self.mindspore_test_model.eval(testset)

        self.pause_training()
        return accuracy['Accuracy']
