"""
A personalized federated learning trainer using Per-FedAvg
"""

from typing import Iterator, Tuple, Union
from collections import OrderedDict
import copy

import torch

from pflbases import personalized_trainer
from plato.config import Config


def get_data_batch(
    dataloader: torch.utils.data.DataLoader,
    iterator: Iterator,
    device=torch.device("cpu"),
):
    """Get one data batch from the dataloader."""
    try:
        samples, labels = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        samples, labels = next(iterator)

    return samples.to(device), labels.to(device)


class Trainer(personalized_trainer.Trainer):
    """A personalized federated learning trainer using the Per-FedAvg algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the iterator for the dataloader
        self.iter_trainloader = None

    def train_epoch_start(self, config):
        """Defining the iterator for the train dataloader."""
        super().train_epoch_start(config)
        self.iter_trainloader = iter(self.train_loader)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        This implementation derives from
        https://github.com/jhoon-oh/FedBABU
        """

        if self.do_final_personalization:
            return self.personalized_forward_and_backward_passes(
                config, examples, labels
            )

        alpha = Config().algorithm.alpha
        beta = Config().algorithm.beta
        temp_net = copy.deepcopy(list(self.model.parameters()))

        # Step 1
        for g in self.optimizer.param_groups:
            g["lr"] = alpha

        self.model.zero_grad()

        logits = self.model(examples)

        loss = self._loss_criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

        # Step 2
        for g in self.optimizer.param_groups:
            g["lr"] = beta

        examples, labels = next(self.iter_trainloader)
        examples, labels = examples.to(self.device), labels.to(self.device)

        self.model.zero_grad()

        logits = self.model(examples)

        loss = self._loss_criterion(logits, labels)
        self._loss_tracker.update(loss, labels.size(0))
        loss.backward()

        # restore the model parameters to the one before first update
        for old_p, new_p in zip(self.model.parameters(), temp_net):
            old_p.data = new_p.data.clone()

        self.optimizer.step()

        return loss

    def personalized_forward_and_backward_passes(self, config, examples, labels):
        """Performing the forward pass for the personalized learning."""

        alpha = Config().algorithm.alpha
        beta = Config().algorithm.beta

        self.personalized_model.train()
        self.personalized_model.to(self.device)

        # Step 1
        for g in self.optimizer.param_groups:
            g["lr"] = alpha

        # Clear the previous gradient
        self.personalized_model.zero_grad()

        # Perfrom the training and compute the loss
        preds = self.personalized_model(examples)
        loss = self._loss_criterion(preds, labels)

        # Perfrom the optimization
        loss.backward()
        self.optimizer.step()

        # Step 2
        # Update the epoch loss container
        for g in self.optimizer.param_groups:
            g["lr"] = beta

        examples, labels = next(self.iter_trainloader)
        examples, labels = examples.to(self.device), labels.to(self.device)

        # Clear the previous gradient
        self.personalized_model.zero_grad()

        # Perfrom the training and compute the loss
        preds = self.personalized_model(examples)
        loss = self._loss_criterion(preds, labels)
        self._loss_tracker.update(loss, labels.size(0))

        # Perfrom the optimization
        loss.backward()
        self.optimizer.step()

        return loss
