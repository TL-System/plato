"""
The training loop for split learning.
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from plato.config import Config

from plato.trainers import basic


class Trainer(basic.Trainer):

    def train_model(self, config, trainset, sampler, cut_layer=None):
        """
        Update the global model on the server with the features received 
        from the client and calculate the gradients.
        """
        batch_size = config["batch_size"]
        log_interval = 10

        if self.client_id == 0:
            logging.info("[Server #%d] Loading the dataset.", os.getpid())
        else:
            logging.info("[Client #%d] Loading the dataset.", self.client_id)

        train_loader = trainset

        iterations_per_epoch = len(train_loader)

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        optimizer = self.get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if hasattr(Config().trainer, "lr_schedule"):
            lr_schedule = self.get_lr_schedule(config, optimizer)
        else:
            lr_schedule = None

        if self.client_id == 0:
            logging.info("[Server #%d] Beginning to train.", os.getpid())
        else:
            logging.info("[Client #%d] Beginning to train.", self.client_id)

        # Record the gradients of the cut layer
        cut_layer_grad = []

        for batch_id, (examples, labels) in enumerate(train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            examples = examples.detach().requires_grad_(True)

            outputs = self.model(examples)

            loss = loss_criterion(outputs, labels)
            loss.backward()

            # Record gradients within the cut layer
            cut_layer_grad.append(examples.grad.clone().detach())

            optimizer.step()

            if batch_id % log_interval == 0:
                if self.client_id == 0:
                    logging.info(
                        "[Server #%d] Batch [%d/%d]\tLoss: %.6f",
                        os.getpid(),
                        batch_id,
                        len(train_loader),
                        loss.data.item(),
                    )
                else:
                    logging.info(
                        "[Client #%d] Batch [%d/%d]\tLoss: %.6f",
                        self.client_id,
                        batch_id,
                        len(train_loader),
                        loss.data.item(),
                    )

            if lr_schedule is not None:
                lr_scheduler.step()

        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

        self.save_gradients(cut_layer_grad)

    def save_gradients(self, cut_layer_grad):
        """Saving gradients to a file."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        if os.path.exists(model_gradients_path):
            os.remove(model_gradients_path)
        torch.save(cut_layer_grad, model_gradients_path)

        logging.info("[Server #%d] Gradients saved to %s.", os.getpid(),
                     model_gradients_path)
