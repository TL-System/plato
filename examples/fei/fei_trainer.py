"""
A customized trainer for FEI.
"""
import logging
import os
import time

import numpy as np
import torch
from opacus.privacy_engine import PrivacyEngine
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """ A federated learning trainer for FEI. """
    def train_model(self, config, trainset, sampler, cut_layer=None):
        """ A custom trainer reporting training loss. """
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

        total_loss = 0
        for epoch in range(1, epochs + 1):
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
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

                total_loss += loss.data.item()

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

        # Save the training loss averaged over epochs in this round
        model_name = config['model_name']
        filename = f'{model_name}_{self.client_id}.loss'
        avg_loss = total_loss / epochs
        Trainer.save_loss(avg_loss, filename)

    @staticmethod
    def save_loss(loss, filename=None):
        """ Saving the training loss to a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            loss_path = f'{model_dir}/{filename}'
        else:
            loss_path = f'{model_dir}/{model_name}.loss'

        with open(loss_path, 'w', encoding='utf-8') as file:
            file.write(str(loss))

    @staticmethod
    def load_loss(filename=None):
        """ Loading the training loss from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if filename is not None:
            loss_path = f'{model_dir}/{filename}'
        else:
            loss_path = f'{model_dir}/{model_name}.loss'

        with open(loss_path, 'r', encoding='utf-8') as file:
            loss = float(file.read())

        return loss
