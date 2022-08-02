"""
A personalized federated learning trainer using Per-FedAvg
"""

import os
import logging
import warnings
from typing import Iterator, Tuple, Union
from collections import OrderedDict
from copy import deepcopy

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers

from plato.utils.checkpoint_operator import perform_client_checkpoint_saving


def get_data_batch(
        dataloader: torch.utils.data.DataLoader,
        iterator: Iterator,
        device=torch.device("cpu"),
):
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        x, y = next(iterator)

    return x.to(device), y.to(device)


def compute_grad(
    model: torch.nn.Module,
    loss_criterion,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
    v: Union[Tuple[torch.Tensor, ...], None] = None,
    second_order_grads=False,
):
    x, y = data_batch
    if second_order_grads:
        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        logit_1 = model(x)
        # loss_1 = loss_criterion(logit_1, y) / y.size(-1)
        loss_1 = loss_criterion(logit_1, y)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        logit_2 = model(x)
        loss_2 = loss_criterion(logit_2, y)
        # loss_2 = loss_criterion(logit_2, y) / y.size(-1)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads, loss_2

    else:
        logit = model(x)
        # loss = loss_criterion(logit, y) / y.size(-1)
        loss = loss_criterion(logit, y)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads, loss


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the Per-FedAvg algorithm."""

    def train_one_epoch(self, config, epoch, defined_model, optimizer,
                        loss_criterion, train_data_loader, epoch_loss_meter,
                        batch_loss_meter):
        defined_model.train()

        is_hessian_free = config['hessian_free']
        alpha = config['alpha']
        beta = config['beta']
        epochs = config['epochs']
        iterations_per_epoch = len(train_data_loader)

        # default not to perform any logging
        epoch_log_interval = epochs + 1
        batch_log_interval = iterations_per_epoch

        if "epoch_log_interval" in config:
            epoch_log_interval = config['epoch_log_interval']
        if "batch_log_interval" in config:
            batch_log_interval = config['batch_log_interval']

        iter_trainloader = iter(train_data_loader)

        epoch_loss_meter.reset()
        # Use a default training loop
        for batch_id, (_, _) in enumerate(train_data_loader):
            # Reset and clear previous data
            batch_loss_meter.reset()
            if is_hessian_free:  # Per-FedAvg(HF)
                temp_model = deepcopy(defined_model)
                data_batch_1 = get_data_batch(train_data_loader,
                                              iter_trainloader, self.device)
                grads, _ = compute_grad(temp_model, loss_criterion,
                                        data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(alpha * grad)

                data_batch_2 = get_data_batch(train_data_loader,
                                              iter_trainloader, self.device)
                grads_1st, _ = compute_grad(temp_model, loss_criterion,
                                            data_batch_2)

                data_batch_3 = get_data_batch(train_data_loader,
                                              iter_trainloader, self.device)

                grads_2nd, loss = compute_grad(defined_model,
                                               loss_criterion,
                                               data_batch_3,
                                               v=grads_1st,
                                               second_order_grads=True)
                batch_size = data_batch_3[1].size(0)
                for param, grad1, grad2 in zip(defined_model.parameters(),
                                               grads_1st, grads_2nd):
                    param.data.sub_(beta * grad1 - beta * alpha * grad2)
            else:
                # Per-FedAvg(FO)
                # ========================== FedAvg ==========================
                # NOTE: You can uncomment those codes for running FedAvg.
                #       When you're trying to run FedAvg, comment other codes in this branch.

                # data_batch = utils.get_data_batch(
                #     self.trainloader, self.iter_trainloader, self.device
                # )
                # grads = self.compute_grad(defined_model, data_batch)
                # for param, grad in zip(defined_model.parameters(), grads):
                #     param.data.sub_(beta * grad)

                # ============================================================

                temp_model = deepcopy(defined_model)
                data_batch_1 = get_data_batch(train_data_loader,
                                              iter_trainloader, self.device)
                grads, _ = compute_grad(temp_model, loss_criterion,
                                        data_batch_1)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(alpha * grad)

                data_batch_2 = get_data_batch(train_data_loader,
                                              iter_trainloader, self.device)
                grads, loss = compute_grad(temp_model, loss_criterion,
                                           data_batch_2)

                batch_size = data_batch_2[1].size(0)
                for param, grad in zip(defined_model.parameters(), grads):
                    param.data.sub_(beta * grad)

            # Update the loss data in the logging container
            epoch_loss_meter.update(loss.data.item(), batch_size)
            batch_loss_meter.update(loss.data.item(), batch_size)

            # Performe logging of one batch
            if batch_id % batch_log_interval == 0 or batch_id == iterations_per_epoch - 1:
                if self.client_id == 0:
                    logging.info(
                        "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                        os.getpid(), epoch, epochs, batch_id,
                        iterations_per_epoch - 1, batch_loss_meter.avg)
                else:
                    logging.info(
                        "   [Client #%d] Training Epoch: \
                        [%d/%d][%d/%d]\tLoss: %.6f", self.client_id, epoch,
                        epochs, batch_id, iterations_per_epoch - 1,
                        batch_loss_meter.avg)

        # Performe logging of epochs
        if (epoch - 1) % epoch_log_interval == 0 or epoch == epochs:
            logging.info("[Client #%d] Training Epoch: [%d/%d]\tLoss: %.6f",
                         self.client_id, epoch, epochs, loss)
