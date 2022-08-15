"""
The training loop for split learning.
"""
import logging
import os

import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)

        # Record the gradients of the cut layer
        self.cut_layer_grad = []

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        batch_size = config["batch_size"]

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
            )

        epochs = config["epochs"]

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer", optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if "lr_schedule" in config:
            lr_schedule = optimizers.get_lr_schedule(
                optimizer, len(train_loader), train_loader
            )
        else:
            lr_schedule = None

        self.model.to(self.device)
        self.model.train()

        for __, (examples, labels) in enumerate(train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            examples = examples.detach().requires_grad_(True)

            outputs = self.model(examples)

            loss = loss_criterion(outputs, labels)

            loss.backward()

            # Record gradients within the cut layer
            self.cut_layer_grad.append(examples.grad.clone().detach())

            optimizer.step()

            if lr_schedule is not None:
                lr_scheduler.step()

        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

        self.save_gradients()

    def save_gradients(self):
        """Saving gradients to a file."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        torch.save(self.cut_layer_grad, model_gradients_path)

        logging.info(
            "[Server #%d] Gradients saved to %s.", os.getpid(), model_gradients_path
        )
