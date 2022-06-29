"""The training loop that takes place on clients"""

from collections import OrderedDict

import os
import logging
import time
import torch

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def train_loop(self, config, trainset, sampler, cut_layer):
        """Saves the loss after each batch pass."""
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

        epochs = config['epochs']

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
        if 'lr_schedule' in config:
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     len(train_loader),
                                                     train_loader)
        else:
            lr_schedule = None

        self.model.to(self.device)
        self.model.train()

        loss_dict = OrderedDict()

        for epoch in range(1, epochs + 1):
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)

                # Track the loss
                if epoch == 1:
                    loss_dict[batch_id] = torch.square(loss)
                else:
                    loss_dict[batch_id] += torch.square(loss)

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

        model_name = Config().trainer.model_name
        model_path = Config().params['checkpoint_path']
        sum_loss = 0
        for batch_id in loss_dict:
            sum_loss += loss_dict[batch_id]
        sum_loss /= epochs
        filename = f"{model_path}/{model_name}_{self.client_id}__squred_batch_loss.pth"
        torch.save(sum_loss, filename)
