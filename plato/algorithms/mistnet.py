"""
The PyTorch-based MistNet algorithm, used by both the client and the server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import time

import torch
from plato.algorithms import fedavg
from plato.datasources import feature_dataset


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based MistNet algorithm, used by both the client and the
    server.
    """

    def extract_features(self, dataset, sampler):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        """
        self.model.eval()

        data_loader = self.trainer.get_train_loader(
            batch_size=1, trainset=dataset, sampler=sampler.get(), extract_features=True
        )

        tic = time.perf_counter()

        features_dataset = []

        for inputs, targets, *__ in data_loader:
            with torch.no_grad():
                logits = self.model.forward_to(inputs)
            features_dataset.append((logits, targets))

        toc = time.perf_counter()
        logging.info("[Client #%s] Time used: %.2f seconds.", self.client_id, toc - tic)

        return features_dataset

    def train(self, trainset, sampler):
        """Train the neural network model after the cut layer."""
        self.trainer.train(
            feature_dataset.FeatureDataset(trainset.feature_dataset), sampler
        )
