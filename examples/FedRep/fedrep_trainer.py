"""
A personalized federated learning trainer using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import os
import time
import logging

import torch
import numpy as np

from opacus.privacy_engine import PrivacyEngine

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def __init__(self, model=None):
        super().__init__(model)

        self.representation_param_names = []
        self.head_param_names = []

    def set_representation_and_head(self, representation_param_names):
        """ Setting the parameter names for global (representation)
         and local (the head) models. """

        # set the parameter names for the representation
        #   As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        #   behaves as the global model.
        self.representation_param_names = representation_param_names

        # FedRep calls the weights and biases of the final fully-connected layer
        # in each of the models as the "head"
        # This insight is obtained from the source code of FedRep.
        model_parameter_names = self.model.state_dict().keys()

        self.head_param_names = [
            name for name in model_parameter_names
            if name not in representation_param_names
        ]

        logging.info("[Client #%s] Representation layers: %s", self.client_id,
                     self.representation_param_names)
        logging.info("[Client #%s] Head layers: %s", self.client_id,
                     self.head_param_names)

    def train_model(self, config, trainset, sampler, cut_layer=None):
        """The main training loop of FedRep in a federated learning workload.

            The local training stage contains two parts:
                - Head optimization:
                Makes τ local gradient-based updates to solve for its optimal head given
                the current global representation communicated by the server.
                - Representation optimization:
                Takes one local gradient-based update with respect to the current representation
        """
        batch_size = config['batch_size']
        log_interval = 10
        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler,
                                             cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=batch_size,
                                                       sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        # load the total local update epochs
        epochs = config['epochs']
        # load the local update epochs for head optimization
        head_epochs = config[
            'head_epochs'] if 'head_epochs' in config else epochs - 1

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if hasattr(config, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        for epoch in range(1, epochs + 1):

            # As presented in the Section 3 of the FedRep paper,
            #   the head is optimized for (epochs - 1) while frozing
            #   the representation
            if epoch <= head_epochs:
                for name, param in self.model.named_parameters():
                    if name in self.representation_param_names:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            # Then, the representation will be optimized for only one
            #   epoch.
            if epoch > head_epochs:
                for name, param in self.model.named_parameters():
                    if name in self.representation_param_names:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())

            if lr_schedule is not None:
                lr_schedule.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if hasattr(Config().server,
                       'request_update') and Config().server.request_update:
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)
