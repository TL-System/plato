"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    def train_model(self, config, trainset, sampler, cut_layer=None):
        """A custom trainer reporting training loss. """
        log_interval = 10
        batch_size = config['batch_size']

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

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
            loss_criterion = nn.CrossEntropyLoss()

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

        try:
            for epoch in range(1, epochs + 1):
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

                    if lr_schedule is not None:
                        lr_schedule.step()

                    if batch_id % log_interval == 0:
                        if self.client_id == 0:
                            logging.info(
                                "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                .format(os.getpid(), epoch, epochs, batch_id,
                                        len(train_loader), loss.data.item()))
                        else:
                            logging.info(
                                "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                .format(self.client_id, epoch, epochs,
                                        batch_id, len(train_loader),
                                        loss.data.item()))

                if hasattr(optimizer, "params_state_update"):
                    optimizer.params_state_update()

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        # Save the training loss of the last epoch in this round
        model_name = config['model_name']
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.loss"
        Trainer.save_loss(loss.data.item(), filename)

    @staticmethod
    def save_loss(loss, filename=None):
        """ Saving the training loss to a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            loss_path = f"{model_dir}{filename}"
        else:
            loss_path = f'{model_dir}{model_name}.loss'

        with open(loss_path, 'w') as file:
            file.write(str(loss))

    @staticmethod
    def load_loss(filename=None):
        """ Loading the training loss from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if filename is not None:
            loss_path = f"{model_dir}{filename}"
        else:
            loss_path = f'{model_dir}{model_name}.loss'

        with open(loss_path, 'r') as file:
            loss = float(file.read())

        return loss
