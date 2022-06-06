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
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based MistNet algorithm, used by both the client and the
    server.
    """
    def extract_features(self, dataset, sampler, cut_layer: str):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        """
        self.model.eval()

        _train_loader = getattr(self.trainer, "train_loader", None)

        if callable(_train_loader):
            data_loader = self.trainer.train_loader(batch_size=1,
                                                    trainset=dataset,
                                                    sampler=sampler.get(),
                                                    extract_features=True)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=Config().trainer.batch_size,
                sampler=sampler.get())

        tic = time.perf_counter()

        feature_dataset = []

        for inputs, targets, *__ in data_loader:
            with torch.no_grad():
                logits = self.model.forward_to(inputs, cut_layer)
            feature_dataset.append((logits, targets))

        toc = time.perf_counter()
        logging.info("[Client #%s] Time used: %.2f seconds.", self.client_id,
                     toc - tic)

        return feature_dataset

    def train(self, trainset, sampler, cut_layer=None):
        """ Train the neural network model after the cut layer. """
        self.trainer.train(
            feature_dataset.FeatureDataset(trainset.feature_dataset), sampler,
            cut_layer)
