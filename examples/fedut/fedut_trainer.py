"""
A federated learning client using Sharpness Aware Minimization.

Reference:

"""
import torch
#from plato.utils import optimizers
from plato.trainers import basic
from sam_optimizer import SAM
import asyncio
import logging
import multiprocessing as mp
import os
import re
import time

import numpy as np
import torch
from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import base
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client. """
    def get_optimizer(self, model):
        base_optimizer = torch.optim.SGD  # optimizers.get_optimizer
        print("Decorating base optimizer as a SAM optimizer")
        return SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                # Use a custom training loop to train for one epoch
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 100
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                _train_loader = getattr(self, "train_loader", None)

                if callable(_train_loader):
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=sampler.get())

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
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
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None

                if 'differential_privacy' in config and config[
                        'differential_privacy']:
                    privacy_engine = PrivacyEngine(accountant='rdp',
                                                   secure_mode=False)

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
                loss_dict = {}
                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(train_loader):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        if 'differential_privacy' in config and config[
                                'differential_privacy']:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            optimizer.zero_grad()

                        if cut_layer is None:
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)
                        # first forward-backward pass

                        loss = loss_criterion(outputs, labels)

                        # track loss info

                        if hasattr(Config().server, 'request_loss') and Config(
                        ).server.request_loss:

                            if epoch == 1:
                                loss_dict[batch_id] = torch.square(loss)
                            else:
                                loss_dict[batch_id] += torch.square(loss)

                        loss.backward(retain_graph=True)
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward pass
                        if cut_layer is None:
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)

                        loss2 = loss_criterion(outputs, labels)
                        loss2.backward()
                        optimizer.second_step(zero_grad=True)

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))

                    if lr_schedule is not None:
                        lr_schedule.step()

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

                    # Simulate client's speed
                    if self.client_id != 0 and hasattr(
                            Config().clients, "speed_simulation") and Config(
                            ).clients.speed_simulation:
                        self.simulate_sleep_time()

                    # Saving the model at the end of this epoch to a file so that
                    # it can later be retrieved to respond to server requests
                    # in asynchronous mode when the wall clock time is simulated

                    if hasattr(Config().server, 'request_update') and Config(
                    ).server.request_update:
                        self.model.cpu()
                        training_time = time.perf_counter() - tic
                        filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                        self.save_model(filename)
                        self.model.to(self.device)

                if hasattr(Config().server,
                           'request_loss') and Config().server.request_loss:
                    sum_loss = 0
                    for batch_id in loss_dict:
                        sum_loss += loss_dict[batch_id]
                    sum_loss /= epochs
                    filename = f"{self.client_id}__squred_batch_loss.pth"
                    torch.save(sum_loss, filename)

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)
