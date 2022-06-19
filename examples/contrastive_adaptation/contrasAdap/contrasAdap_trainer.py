"""
Implementation of our contrastive adaptation trainer.

"""

import os
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from plato.config import Config
from plato.trainers import contrastive_ssl
from plato.utils import data_loaders_wrapper
from plato.utils import optimizers

from contrasAdap_losses import ContrasAdapLoss


def mean_squared_error(x, y):
    """ Compute the mean square error. """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def loss_fn_with_stop_gradients(outputs):
    """ Compute the errors with stop gradients mechanism. """
    (online_pred_one, online_pred_two), (target_proj_one,
                                         target_proj_two) = outputs

    # use the detach mechanism to stop the gradient for target learner
    loss_one = mean_squared_error(online_pred_one, target_proj_two.detach())
    loss_two = mean_squared_error(online_pred_two, target_proj_one.detach())

    loss = loss_one + loss_two
    return loss.mean()


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the BYOL client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation.
        """

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            loss = loss_fn_with_stop_gradients(outputs)
            return loss

        return loss_compute

    def train_loop(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

        Note:
            This is the training stage of self-supervised learning (ssl). It is responsible
        for performing the contrastive learning process based on the trainset to train
        a encoder in the unsupervised manner. Then, this trained encoder is desired to
        use the strong backbone by downstream tasks to solve the objectives effectively.
            Therefore, the train loop here utilize the
            - trainset with one specific transform (contrastive data augmentation)
            - self.model, the ssl method to be trained.

        """
        batch_size = config['batch_size']

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

        # obtain the loader for unlabeledset if possible
        # unlabeled_trainset, unlabeled_sampler
        unlabeled_loader = None
        unlabeled_trainset = []
        if "unlabeled_trainset" in kwargs:
            unlabeled_trainset = kwargs["unlabeled_trainset"]
            unlabeled_sampler = kwargs["unlabeled_sampler"]
            unlabeled_loader = torch.utils.data.DataLoader(
                dataset=unlabeled_trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=unlabeled_sampler)

        # wrap the multiple loaders into one sequence loader
        train_loader = data_loaders_wrapper.StreamBatchesLoader(
            [train_loader, unlabeled_loader])

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        iterations_per_epoch += np.ceil(len(unlabeled_trainset) /
                                        batch_size).astype(int)

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        loss_criterion = self.loss_criterion(self.model)

        # Initializing the optimizer
        optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if 'lr_schedule' in config:
            lr_schedule = optimizers.get_dynamic_lr_schedule(
                optimizer, iterations_per_epoch, train_loader)
        else:
            lr_schedule = None

        # Obtain the logging interval
        epochs = config['epochs']
        epoch_log_interval = config['epoch_log_interval']
        batch_log_interval = config['batch_log_interval']

        # Define the container to hold the logging information
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        # Start training
        for epoch in range(1, epochs + 1):
            epoch_loss_meter.reset()
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                # Support a more general way to hold the loaded samples
                # The defined model is responsible for processing the
                # examples based on its requirements.
                if torch.is_tensor(examples):
                    examples = examples.to(self.device)
                else:
                    examples = [
                        each_sample.to(self.device) for each_sample in examples
                    ]

                labels = labels.to(self.device)

                # Reset and clear previous data
                batch_loss_meter.reset()
                optimizer.zero_grad()

                # Forward the model and compute the loss
                outputs = self.model(examples)
                loss = loss_criterion(outputs, labels)

                # Perform the backpropagation
                loss.backward()
                optimizer.step()

                # Update the loss data in the logging container
                epoch_loss_meter.update(loss.data.item())
                batch_loss_meter.update(loss.data.item())

                # Performe logging of batches
                if batch_id % batch_log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), batch_loss_meter.avg)
                    else:
                        logging.info(
                            "   [Client #%d] Contrastive Pre-train Epoch: \
                            [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(train_loader), batch_loss_meter.avg)

            # Performe logging of epochs
            if epoch - 1 % epoch_log_interval == 0:
                logging.info(
                    "[Client #%d] Contrastive Pre-train Epoch: [%d/%d]\tLoss: %.6f",
                    self.client_id, epoch, epochs, epoch_loss_meter.avg)

            # Update the learning rate
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
