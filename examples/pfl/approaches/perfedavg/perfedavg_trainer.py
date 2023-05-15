"""
A personalized federated learning trainer using Per-FedAvg
"""

import os
from typing import Iterator, Tuple, Union
from collections import OrderedDict
import copy

import torch

from plato.utils.filename_formatter import NameFormatter

from pflbases import personalized_trainer


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


def compute_gradients(
    model: torch.nn.Module,
    loss_criterion,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
    base_grads: Union[Tuple[torch.Tensor, ...], None] = None,
    second_order_grads=False,
):
    """Compute the gradients."""
    examples, labels = data_batch
    if second_order_grads:
        frz_model_params = copy.deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), base_grads):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        logit_1 = model(examples)
        # loss_1 = loss_criterion(logit_1, y) / y.size(-1)
        loss_1 = loss_criterion(logit_1, labels)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        logit_2 = model(examples)
        loss_2 = loss_criterion(logit_2, labels)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads, loss_2

    else:
        logit = model(examples)
        # loss = loss_criterion(logit, y) / y.size(-1)
        loss = loss_criterion(logit, labels)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads, loss


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

    def meta_forward_and_backward_passes_V2(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        This implementation derives from
        https://github.com/KarhouTam/Per-FedAvg
        """
        alpha = config["alpha"]
        beta = config["beta"]
        if config["hessian_free"]:  # Per-FedAvg(HF)
            temp_model = copy.deepcopy(self.model)

            grads, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch=(examples, labels)
            )
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(alpha * grad)

            data_batch_2 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )
            grads_1st, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch_2
            )

            data_batch_3 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )
            grads_2nd, loss = compute_gradients(
                self.model,
                self._loss_criterion,
                data_batch_3,
                base_grads=grads_1st,
                second_order_grads=True,
            )

            for param, grad1, grad2 in zip(
                self.model.parameters(), grads_1st, grads_2nd
            ):
                param.data.sub_(beta * grad1 - beta * alpha * grad2)

        else:  # Per-FedAvg(FO)
            temp_model = copy.deepcopy(self.model)
            grads, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch=(examples, labels)
            )
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(alpha * grad)

            data_batch_2 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )
            grads, loss = compute_gradients(
                temp_model, self._loss_criterion, data_batch_2
            )
            for param, grad in zip(self.model.parameters(), grads):
                param.data.sub_(beta * grad)

        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        This implementation derives from
        https://github.com/jhoon-oh/FedBABU
        """
        alpha = config["alpha"]
        beta = config["beta"]
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

    def meta_forward_and_backward_passes(self, config, examples, labels):
        """Performing the meta training.        This implementation derives from
        https://github.com/jhoon-oh/FedBABU
        """
        alpha = config["alpha"]
        beta = config["beta"]
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

        alpha = config["alpha"]
        beta = config["beta"]

        self.personalized_model.train()
        self.personalized_model.to(self.device)

        # Step 1
        for g in self.personalized_optimizer.param_groups:
            g["lr"] = alpha
        examples, labels = next(self.iter_personalized_trainloader)
        examples, labels = examples.to(self.device), labels.to(self.device)

        # Clear the previous gradient
        self.personalized_model.zero_grad()

        # Perfrom the training and compute the loss
        preds = self.personalized_model(examples)
        loss = self._loss_criterion(preds, labels)

        # Perfrom the optimization
        loss.backward()
        self.personalized_optimizer.step()

        # Step 2
        # Update the epoch loss container
        for g in self.personalized_optimizer.param_groups:
            g["lr"] = beta

        examples, labels = next(self.iter_personalized_trainloader)
        examples, labels = examples.to(self.device), labels.to(self.device)

        # Clear the previous gradient
        self.personalized_model.zero_grad()

        # Perfrom the training and compute the loss
        preds = self.personalized_model(examples)
        loss = self._personalized_loss_criterion(preds, labels)

        # Perfrom the optimization
        loss.backward()
        self.personalized_optimizer.step()

        return loss
