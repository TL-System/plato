"""
The federated learning trainer for MistNet, used by both the client and the
server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import time
import logging
import mindspore
import mindspore.dataset as ds

from utils import unary_encoding
from trainers import trainer_mindspore

class Trainer(trainer_mindspore.Trainer):
    """A federated learning trainer for MistNet in the MindSpore framework, used
    by both the client and the server.
    """
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

            logit = mindspore.Tensor(logits.asnumpy())
            target = mindspore.Tensor(targets.asnumpy())
            feature_dataset.append((logit, target))

        toc = time.perf_counter()
        logging.info("[Client #%s] Features extracted from %s examples.",
            self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    @staticmethod
    def dataset_generator(trainset):
        """The generator used to produce a suitable Dataset for the MineSpore trainer."""
        for logit, target in trainset:
            yield logit.asnumpy(), target.asnumpy()

    def train(self, trainset, cut_layer=None):
        """The main training loop used in the MistNet server.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        feature_dataset = ds.GeneratorDataset(
            list(Trainer.dataset_generator(trainset)), column_names=["image", "label"])

        super().train(feature_dataset, cut_layer)

    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        self.start_training()

        self.model.cut_layer = None
        accuracy = self.mindspore_model.eval(testset)

        self.pause_training()
        return accuracy['Accuracy']
