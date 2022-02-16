"""
The training and testing loops for PyTorch.
"""
import copy
import logging
import os

import numpy as np
import torch

import subfedavg_pruning as pruning_processor
from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A federated learning trainer for Sub-Fedavg algorithm."""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)
        self.mask = None
        self.pruning_amount = Config().clients.pruning_amount * 100
        self.made_init_mask = False

    def train_loop(self, config, trainset, sampler, cut_layer):
        """ The default training loop when a custom training loop is not supplied. """
        batch_size = config['batch_size']
        log_interval = 10

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

        if not self.made_init_mask:
            self.mask = pruning_processor.make_init_mask(self.model)
            self.made_init_mask = True

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

                # Freezing Pruned weights by making their gradients Zero
                step = 0
                for name, parameter in self.model.named_parameters():
                    if 'weight' in name:
                        grad_tensor = parameter.grad.data.cpu().numpy()
                        grad_tensor = grad_tensor * self.mask[step]
                        parameter.grad.data = torch.from_numpy(grad_tensor).to(
                            self.device)
                        step = step + 1

                optimizer.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())

            if lr_schedule is not None:
                lr_schedule.step()

            if epoch == 1:
                first_epoch_mask = pruning_processor.fake_prune(
                    self.pruning_amount, copy.deepcopy(self.model),
                    copy.deepcopy(self.mask))
            if epoch == epochs:
                last_epoch_mask = pruning_processor.fake_prune(
                    self.pruning_amount, copy.deepcopy(self.model),
                    copy.deepcopy(self.mask))

        mask_distance = pruning_processor.dist_masks(first_epoch_mask,
                                                     last_epoch_mask)
        mask_distance_threshold = Config(
        ).clients.mask_distance_threshold if hasattr(
            Config().clients, "mask_distance_threshold") else 0.0001

        if mask_distance > mask_distance_threshold:
            last_epoch_mask = pruning_processor.fake_prune(
                self.pruning_amount, copy.deepcopy(self.model),
                copy.deepcopy(self.mask))

            pruned_weights = pruning_processor.real_prune(
                copy.deepcopy(self.model), last_epoch_mask)
            self.model.load_state_dict(pruned_weights, strict=True)

        self.mask = copy.deepcopy(last_epoch_mask)
