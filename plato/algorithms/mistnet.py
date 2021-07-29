"""
The PyTorch-based MistNet algorithm, used by both the client and the server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import time

import numpy as np
import torch
from plato.config import Config
from plato.utils import unary_encoding

from plato.algorithms import fedavg


class FeatureDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based MistNet algorithm, used by both the client and the
    server.
    """
    def extract_features(self, dataset, sampler, cut_layer: str, epsilon=None):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        epsilon: If epsilon is not None, local differential privacy should be
                applied to the features extracted.
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

        _randomize = getattr(self.trainer, "randomize", None)

        for inputs, targets, *__ in data_loader:
            with torch.no_grad():
                logits = self.model.forward_to(inputs, cut_layer)
                if epsilon is not None:
                    logits = logits.detach().numpy()
                    logits = unary_encoding.encode(logits)
                    if callable(_randomize):
                        logits = self.trainer.randomize(
                            logits, targets, epsilon)
                    else:
                        logits = unary_encoding.randomize(logits, epsilon)
                    if self.trainer.device != 'cpu':
                        logits = torch.from_numpy(logits.astype('float16'))
                    else:
                        logits = torch.from_numpy(logits.astype('float32'))

            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))

        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    def train(self, trainset, sampler, cut_layer=None):
        """ Train the neural network model after the cut layer. """
        self.trainer.train(FeatureDataset(trainset), sampler, cut_layer)
