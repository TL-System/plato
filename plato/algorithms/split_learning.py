"""
A federated learning algorithm using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""

import logging
import time

from plato.algorithms import fedavg
from plato.config import Config
from plato.datasources import feature_dataset


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based split learning algorithm, used by both the client and the
    server.
    """

    def extract_features(self, dataset, sampler):
        """Extracting features using layers before the cut_layer."""
        self.model.to(self.trainer.device)
        self.model.eval()

        tic = time.perf_counter()

        features_dataset = []

        inputs, targets = self.trainer.get_train_samples(
            Config().trainer.batch_size, dataset, sampler
        )
        inputs = inputs.to(self.trainer.device)
        targets = targets.to(self.trainer.device)
        outputs, targets = self.trainer.forward_to_intermediate_feature(inputs, targets)
        features_dataset.append((outputs, targets))

        toc = time.perf_counter()
        logging.warning(
            "[Client #%d] Features extracted from %s examples in %.2f seconds.",
            self.client_id,
            Config().trainer.batch_size,
            toc - tic,
        )

        return features_dataset, toc - tic

    def complete_train(self, gradients):
        """Update the model on the client/device with the gradients received
        from the server.
        """
        tic = time.perf_counter()

        # Retrieve the training samples and let trainer do the training
        samples, sampler = self.trainer.retrieve_train_samples()
        self.trainer.load_gradients(gradients)
        self.train(samples, sampler)

        toc = time.perf_counter()
        logging.warning(
            "[Client #%d] Training completed in %.2f seconds.",
            self.client_id,
            toc - tic,
        )

        return toc - tic

    def train(self, trainset, sampler):
        """General training method that trains model with provided trainset and sampler."""
        self.trainer.train(
            feature_dataset.FeatureDataset(trainset.feature_dataset), sampler
        )

    def update_weights_before_cut(self, weights):
        """Update the weights before cut layer, called when testing accuracy."""
        current_weights = self.extract_weights()
        current_weights = self.trainer.update_weights_before_cut(
            current_weights, weights
        )
        self.load_weights(current_weights)