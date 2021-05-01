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
from config import Config
from utils import optimizers

from trainers import basic

class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)

        # Record the gradients w.r.t cut layer in a batch
        self.grad_input_cus = None
        self.grad_output_cus = None

        # Record the gradients w.r.t whole batches
        self.gradients_list = []

    def save_gradients(self, module, grad_input, grad_output):
        """
        Use to record gradients
        Called by register_backward_hook
        """
        self.grad_input_cus = grad_input
        self.grad_output_cus = grad_output

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument

        self.gradients_list.clear()

        if cut_layer is not None and hasattr(self.model, cut_layer):
            # Fine the layer next to cut_layer
            cut_layer_index = self.model.layers.index(cut_layer)
            if cut_layer_index < (len(self.model.layers) - 1):
                hook_layer = self.model.layers[cut_layer_index + 1]
            else:
                hook_layer = cut_layer
            self.model.layerdict[hook_layer].register_backward_hook(self.save_gradients)

        log_interval = 10
        batch_size = config['batch_size']

        logging.info("[Client #%s] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = _train_loader(batch_size, trainset,
                                            sampler, cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) /
                                        batch_size).astype(int)

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
            lr_schedule = optimizers.get_lr_schedule(
                optimizer, iterations_per_epoch, train_loader)
        else:
            lr_schedule = None
        
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

            # Record gradients in this batch
            if (self.grad_input_cus is not None
                ) and (self.grad_output_cus is not None):
                self.gradients_list.append(self.grad_input_cus)
                self.gradients_list.append(self.grad_output_cus)
            self.grad_input_cus = None
            self.grad_output_cus = None

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

        torch.save(self.gradients_list, model_path)

        logging.info("[Server #%s] Gradients saved to %s.", os.getpid(),
                    model_path)