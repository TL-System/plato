"""
A federated learning algorithm using split learning.

Reference:

Vepakomma, et al., "Split learning for health: Distributed deep learning without sharing
raw patient data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf
"""

import logging
import time
from copy import deepcopy

import torch

from plato.algorithms import fedavg
from plato.config import Config
from plato.datasources import feature_dataset


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based split learning algorithm, used by both the client and the
    server.
    """

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.gradients_list = []
        self.input_dataset = []
        self.data_loader = None


    def receive_gradients(self, gradients):
        """Receive gradients from the server."""
        self.gradients_list = deepcopy(gradients)

    def extract_features(self):
        """Extracting features using layers before the cut_layer."""
        self.model.to(self.trainer.device)
        self.model.eval()

        tic = time.perf_counter()

        features_dataset = []
        self.input_dataset = []

        inputs, targets = next(self.data_loader)
        with torch.no_grad():
            inputs, targets = inputs.to(self.trainer.device), targets.to(
                self.trainer.device
            )
            logits = self.model.forward_to(inputs)

        features_dataset.append((logits.detach().cpu(), targets.detach().cpu()))
        self.input_dataset.append((inputs.detach().cpu(), targets.detach().cpu()))

        toc = time.perf_counter()
        logging.warning(
            "[Client #%d] Features extracted from %s examples in %.2f seconds.",
            self.client_id,
            Config().trainer.batch_size,
            toc - tic,
        )

        return features_dataset, toc - tic

    def complete_train(self):
        """Update the model on the client/device with the gradients received
        from the server.
        """
        self.model.to(self.trainer.device)
        self.model.train()

        data_loader = self.input_dataset

        tic = time.perf_counter()

        # Initializing the optimizer
        optimizer = self.trainer.get_optimizer(self.model)

        grad_index = 0

        for __, (examples, labels) in enumerate(data_loader):
            examples, labels = examples.to(self.trainer.device), labels.to(
                self.trainer.device
            )

            optimizer.zero_grad()
            outputs = self.model.forward_to(examples)
            outputs.backward(self.gradients_list[grad_index].to(self.trainer.device))
            grad_index = grad_index + 1
            optimizer.step()

        toc = time.perf_counter()
        logging.warning(
            "[Client #%d] Training completed in %.2f seconds.",
            self.client_id,
            toc - tic,
        )

        return toc - tic

    def train(self, trainset, sampler):
        """Train the neural network model after the cut layer."""
        self.trainer.train(
            feature_dataset.FeatureDataset(trainset.feature_dataset), sampler
        )

    def load_data(self, dataset, sampler):
        """Setting up the data loader."""
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=Config().trainer.batch_size,
            sampler=sampler.get(),
        )
        self.data_loader = iter(data_loader)

    def update_weights_before_cut(self, weights):
        # Update the weights before cut layer, called when testing accuracy
        current_weights = self.extract_weights()
        cut_layer_idx = self.model.layers.index(self.model.cut_layer)
        for i in range(0, cut_layer_idx):
            weight_name = f"{self.model.layers[i]}.weight"
            bias_name = f"{self.model.layers[i]}.bias"

            if weight_name in current_weights:
                current_weights[weight_name] = weights[weight_name]

            if bias_name in current_weights:
                current_weights[bias_name] = weights[bias_name]
        self.load_weights(current_weights)
