"""
A personalized federated learning trainer using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import os
import time
import logging

import torch
from tqdm import tqdm

from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers
from plato.utils.checkpoint_operator import perform_client_checkpoint_saving

from plato.utils import data_loaders_wrapper


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def freeze_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = False

    def active_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = True

    def train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default personalized training loop when a custom training loop is not supplied.

        """

        batch_size = config['batch_size']
        model_type = config['model_name']
        current_round = kwargs['current_round']
        run_id = config['run_id']
        # Obtain the logging interval
        epochs = config['epochs']

        # load the local update epochs for head optimization
        head_epochs = config[
            'head_epochs'] if 'head_epochs' in config else epochs - 1

        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        # to get the specific sampler, Plato's sampler should perform
        # Sampler.get()
        # However, for train's ssampler, the self.sampler.get() has been
        # performed within the train_process of the trainer/basic.py
        # Thus, there is no need to further perform .get() here.
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   sampler=sampler)

        # obtain the loader for unlabeledset if possible
        # unlabeled_trainset, unlabeled_sampler
        unlabeled_loader = None
        unlabeled_trainset = []
        if "unlabeled_trainset" in kwargs and kwargs[
                "unlabeled_trainset"] is not None:
            unlabeled_trainset = kwargs["unlabeled_trainset"]
            unlabeled_sampler = kwargs["unlabeled_sampler"]
            unlabeled_loader = torch.utils.data.DataLoader(
                dataset=unlabeled_trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=unlabeled_sampler.get())

        # wrap the multiple loaders into one sequence loader
        streamed_train_loader = data_loaders_wrapper.StreamBatchesLoader(
            [train_loader, unlabeled_loader])

        epoch_model_log_interval = epochs + 1
        if "epoch_model_log_interval" in config:
            epoch_model_log_interval = config['epoch_model_log_interval']

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        lr_schedule, lr_schedule_base_epoch = self.prepare_train_lr(
            optimizer, streamed_train_loader, config, current_round)

        # Before the training, we expect to save the initial
        # model of this round
        perform_client_checkpoint_saving(
            client_id=self.client_id,
            model_name=model_type,
            model_state_dict=self.model.state_dict(),
            config=config,
            kwargs=kwargs,
            optimizer_state_dict=optimizer.state_dict(),
            lr_schedule_state_dict=lr_schedule.state_dict(),
            present_epoch=0,
            base_epoch=lr_schedule_base_epoch)

        # Sending the model to the device used for training
        self.model.to(self.device)

        # Define the container to hold the logging information
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        # Start training
        for epoch in range(1, epochs + 1):
            self.model.train()
            # As presented in Section 3 of the FedRep paper, the head is optimized
            # for (epochs - 1) while freezing the representation.
            if epoch <= head_epochs:
                self.freeze_model(self.model, param_prefix="encoder")
                self.active_model(self.model, param_prefix="clf_fc")

            # The representation will then be optimized for only one epoch
            if epoch > head_epochs:
                self.freeze_model(self.model, param_prefix="clf_fc")
                self.active_model(self.model, param_prefix="encoder")

            self.train_one_epoch(config,
                                 epoch,
                                 defined_model=self.model,
                                 optimizer=optimizer,
                                 loss_criterion=loss_criterion,
                                 train_data_loader=streamed_train_loader,
                                 epoch_loss_meter=epoch_loss_meter,
                                 batch_loss_meter=batch_loss_meter)

            # Update the learning rate
            # based on the base epoch
            lr_schedule.step()

            if (epoch - 1) % epoch_model_log_interval == 0 or epoch == epochs:
                # the model generated during each round will be stored in the
                # checkpoints
                perform_client_checkpoint_saving(
                    client_id=self.client_id,
                    model_name=model_type,
                    model_state_dict=self.model.state_dict(),
                    config=config,
                    kwargs=kwargs,
                    optimizer_state_dict=optimizer.state_dict(),
                    lr_schedule_state_dict=lr_schedule.state_dict(),
                    present_epoch=epoch,
                    base_epoch=lr_schedule_base_epoch + epoch)

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

        if 'max_concurrency' in config:
            # the final of each round, the trained model within this round
            # will be saved as model to the '/models' dir

            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=model_type,
                model_state_dict=self.model.state_dict(),
                config=config,
                kwargs=kwargs,
                optimizer_state_dict=optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=None,
                base_epoch=lr_schedule_base_epoch + epochs)