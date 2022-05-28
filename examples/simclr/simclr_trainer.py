"""
Implement the trainer for SimCLR method.

"""

import os
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from opacus.privacy_engine import PrivacyEngine

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers

from nt_xent import NT_Xent


class Trainer(basic.Trainer):

    def __init__(self, model=None):
        super().__init__(model)

    def loss_criterion(self, model):
        """ The loss computation. """
        # define the loss computation instance
        defined_temperature = Config().trainer.temperature
        batch_size = Config().trainer.batch_size

        def loss_compute(outputs, labels):
            z1, z2 = outputs
            criterion = NT_Xent(batch_size, defined_temperature, world_size=1)
            loss = criterion(z1, z2)
            return loss

        return loss_compute

    def train_loop(self, config, trainset, sampler, cut_layer):
        """ The default training loop when a custom training loop is not supplied. """
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
        epochs = config['epochs']

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

        if 'differential_privacy' in config and config['differential_privacy']:
            privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=False)

            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=config['dp_epsilon']
                if 'dp_epsilon' in config else 10.0,
                target_delta=config['dp_delta']
                if 'dp_delta' in config else 1e-5,
                epochs=epochs,
                max_grad_norm=config['dp_max_grad_norm']
                if 'max_grad_norm' in config else 1.0,
            )

        for epoch in range(1, epochs + 1):
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples1, examples2 = examples
                examples1, examples2, labels = examples1.to(
                    self.device), examples2.to(self.device), labels.to(
                        self.device)

                if 'differential_privacy' in config and config[
                        'differential_privacy']:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples1, examples2)
                else:
                    outputs = self.model.forward_from(examples1, examples2,
                                                      cut_layer)

                loss = loss_criterion(outputs, labels)

                if 'create_graph' in config:
                    loss.backward(create_graph=config['create_graph'])
                else:
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