"""
The PyTorch-based split-learning algorithm, used by both the client and the server.
"""

import logging
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from plato.config import Config
from plato.utils import unary_encoding
from plato.utils import optimizers

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
    """The PyTorch-based split learning algorithm, used by both the client and the
    server.
    """
    def __init__(self, trainer=None, client_id=None):
        super().__init__(trainer, client_id)
        self.gradients_list = []

    def load_gradients(self, gradients):
        """
        Receive gradients from server
        """
        self.gradients_list = deepcopy(gradients)

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

        _randomize = getattr(self.trainer, "randomize", None)

        for inputs, targets, *__ in data_loader:
            logits = self.model.forward_to(inputs, cut_layer)

            logits = logits.clone().detach().requires_grad_(True)

            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))

        toc = time.perf_counter()
        logging.info("[Client #%s] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    def complete_train(self, config, dataset, sampler, cut_layer: str):

        # Sending the model to the device used for training
        self.model.train()

        batch_size = config['batch_size']

        _train_loader = getattr(self.trainer, "train_loader", None)

        if callable(_train_loader):
            data_loader = self.trainer.train_loader(batch_size=batch_size,
                                                    trainset=dataset,
                                                    sampler=sampler.get(),
                                                    extract_features=True)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      shuffle=False,
                                                      batch_size=batch_size,
                                                      sampler=sampler.get())

        tic = time.perf_counter()

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = _loss_criterion(self.model)
        else:
            loss_criterion = nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        grad_index = 0

        for batch_id, (examples, labels) in enumerate(data_loader):
            optimizer.zero_grad()

            outputs = self.model.forward_to(examples, cut_layer)

            outputs.backward(self.gradients_list[grad_index])

            grad_index = grad_index + 1

            optimizer.step()

        toc = time.perf_counter()
        # logging.info("[Client #%s] Features extracted from %s examples.",
        #              self.client_id, len(feature_dataset))
        # logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
        #     self.client_id, toc - tic))

    def train(self, trainset, sampler, cut_layer=None):
        self.trainer.train(FeatureDataset(trainset), sampler, cut_layer)
