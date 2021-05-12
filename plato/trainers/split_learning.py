"""
The training loop for split learning.
"""
import logging
import os

import copy
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from plato.config import Config
from plato.utils import optimizers

from plato.trainers import basic


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)

        # Record the gradients w.r.t cut layer in a batch
        self.cut_layer_grad = []

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument

        log_interval = 10
        batch_size = config['batch_size']

        logging.info("[Client #%s] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = _train_loader(batch_size, trainset, sampler,
                                         cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=batch_size,
                                                       sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

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

        # Initializing the learning rate schedule, if necessary
        if hasattr(Config().trainer, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        for batch_id, (examples, labels) in enumerate(train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            examples = examples.detach().requires_grad_(True)

            if cut_layer is None:
                outputs = self.model(examples)
            else:
                outputs = self.model.forward_from(examples, cut_layer)

            loss = loss_criterion(outputs, labels)

            loss.backward()

            #Record gradients of cut_layer
            self.cut_layer_grad.append(examples.grad.clone().detach())

            optimizer.step()

            if lr_schedule is not None:
                lr_schedule.step()

            # if batch_id % log_interval == 0:
            #     if self.client_id == 0:
            #         logging.info(
            #             "[Server #{}] Batch: [{}/{}]\tLoss: {:.6f}"
            #             .format(os.getpid(), batch_id,
            #                     len(train_loader), loss.data.item()))
            #     else:
            #         wandb.log({"batch loss": loss.data.item()})

            #         logging.info(
            #             "[Client #{}] Batch: [{}/{}]\tLoss: {:.6f}"
            #             .format(self.client_id, batch_id,
            #                     len(train_loader),
            #                     loss.data.item()))
        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

        self.save_gradients_to_disk()

    def save_gradients_to_disk(self, filename=None):
        """
        Saving gradients to a file.
        """
        model_name = Config().trainer.model_name
        model_dir = Config().params['model_dir']

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}_gradients.pth'

        torch.save(self.cut_layer_grad, model_path)

        logging.info("[Server #%s] Gradients saved to %s.", os.getpid(),
                     model_path)
