"""
The PyTorch-based split-learning algorithm, used by both the client and the server.
"""

import logging
import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from plato.algorithms import fedavg
from plato.config import Config
from plato.datasources import feature_dataset
from plato.utils import optimizers


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based split learning algorithm, used by both the client and the
    server.
    """

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.gradients_list = []
        self.input_dataset = []

    def receive_gradients(self, gradients):
        """
        Receive gradients from the server.
        """
        self.gradients_list = deepcopy(gradients)

    def extract_features(self, dataset, sampler, cut_layer: str):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        """
        self.model.to(self.trainer.device)
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

        features_dataset = []
        self.input_dataset = []

        for inputs, targets, *__ in data_loader:
            inputs, targets = inputs.to(self.trainer.device), targets.to(
                self.trainer.device)
            logits = self.model.forward_to(inputs, cut_layer)

            logits = logits.clone().detach().requires_grad_(True)
            examples = inputs.clone().detach().requires_grad_(True)

            for i in np.arange(logits.shape[0]):  # each sample in the batch
                features_dataset.append((logits[i], targets[i]))
                # remember the order of the input train data s.t. we can
                # update the weights with gradients from the server in the
                # correct order.
                self.input_dataset.append((examples[i], targets[i]))

        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(features_dataset))
        logging.info("[Client #%d] Time used: %.2f seconds.", self.client_id,
                     toc - tic)

        return features_dataset, toc - tic

    def complete_train(self, config, dataset, sampler, cut_layer: str):
        """ Sending the model to the device used for training. """
        self.model.to(self.trainer.device)
        self.model.train()

        batch_size = config['batch_size']

        _train_loader = getattr(self.trainer, "train_loader", None)

        if callable(_train_loader):
            data_loader = self.trainer.train_loader(
                batch_size=batch_size,
                trainset=self.input_dataset,
                extract_features=True)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset=self.input_dataset,
                shuffle=False,
                batch_size=batch_size)

        tic = time.perf_counter()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        grad_index = 0

        for batch_id, (examples, labels) in enumerate(data_loader):
            examples, labels = examples.to(self.trainer.device), labels.to(
                self.trainer.device)

            optimizer.zero_grad()
            outputs = self.model.forward_to(examples, cut_layer)
            outputs.backward(self.gradients_list[grad_index].to(
                self.trainer.device))
            grad_index = grad_index + 1
            optimizer.step()

        toc = time.perf_counter()

        logging.info("[Client #%d] Training completed.", self.client_id)
        logging.info("[Client #%d] Time used: %.2f seconds.", self.client_id,
                     toc - tic)

        return toc - tic

    def compute_weight_deltas(self, weights_received):
        """Extract the weights received from a client and compute the deltas."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        ignored_layers = []
        cut_layer_idx = self.model.layers.index(Config().algorithm.cut_layer)
        # These layers are trained on the server, so we should ignore the weights
        # of these layers reported by the client
        for i in range(cut_layer_idx + 1, len(self.model.layers)):
            ignored_layers.append(f"{self.model.layers[i]}.weight")
            ignored_layers.append(f"{self.model.layers[i]}.bias")

        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]
                if name in ignored_layers:
                    # Do not update the layers that are not trained on clients
                    _delta = torch.zeros(baseline.shape)
                else:
                    # Calculate update
                    _delta = current_weight - baseline
                delta[name] = _delta
            deltas.append(delta)
        return deltas

    def train(self, trainset, sampler, cut_layer=None):
        self.trainer.train(feature_dataset.FeatureDataset(trainset), sampler,
                           cut_layer)
