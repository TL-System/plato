"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new trainer callbacks.

Defines a default callback to print training progress.
"""
import logging
import os
from abc import ABC

import torch
from plato.utils import fonts


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new trainer callbacks.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """
        Event called at the start of training run.
        """

    def on_train_run_end(self, trainer, config, **kwargs):
        """
        Event called at the end of training run.
        """

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """

    def on_train_step_start(self, trainer, config, batch, **kwargs):
        """
        Event called at the beginning of a training step.

        :param batch: the current batch of training data.
        """

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """
        Event called at the end of a training step.

        :param batch: the current batch of training data.
        :param loss: the loss computed in the current batch.
        """

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """
        Event called at the end of a training epoch.
        """


class LogProgressCallback(TrainerCallback):
    """
    A callback which prints a message at the start of each epoch, and at the end of each step.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """
        Event called at the start of training run.
        """
        if trainer.client_id == 0:
            logging.info(
                "[Server #%s] Loading the dataset with size %d.",
                os.getpid(),
                len(list(trainer.sampler)),
            )
        else:
            logging.info(
                "[Client #%d] Loading the dataset with size %d.",
                trainer.client_id,
                len(list(trainer.sampler)),
            )

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """
        if trainer.client_id == 0:
            logging.info(
                fonts.colourize(
                    f"[Server #{os.getpid()}] Started training epoch {trainer.current_epoch}."
                )
            )
        else:
            logging.info(
                fonts.colourize(
                    f"[Client #{trainer.client_id}] Started training epoch {trainer.current_epoch}."
                )
            )

    def on_train_step_end(self, trainer, config, batch=None, loss=None, **kwargs):
        """
        Event called at the end of a training step.

        :param batch: the current batch of training data.
        :param loss: the loss computed in the current batch.
        """
        log_interval = 10

        if batch % log_interval == 0:
            if trainer.client_id == 0:
                logging.info(
                    "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                    os.getpid(),
                    trainer.current_epoch,
                    config["epochs"],
                    batch,
                    len(trainer.train_loader),
                    loss.data.item(),
                )
            else:
                logging.info(
                    "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                    trainer.client_id,
                    trainer.current_epoch,
                    config["epochs"],
                    batch,
                    len(trainer.train_loader),
                    loss.data.item(),
                )


class SplitLearningCallback(LogProgressCallback):
    """A callback for split learning handling model specific operations."""

    def on_retrieve_train_samples(self, trainer):
        """The event befor retrieviing tianing samples."""

    def on_client_forward_to(self, trainer):
        """The event befor client conducting forwarding."""

    def on_server_forward_from(self, trainer, loss_criterion, input_target_pair):
        "Hook the rules of training on the server to the trainer.model."

        inputs, target = input_target_pair
        inputs = inputs.detach().requires_grad_(True)
        outputs = trainer.model.forward_from(inputs)
        loss = loss_criterion(outputs, target)
        loss.backward()
        grad = inputs.grad
        trainer.loss_grad_pair = (loss, grad)

    def on_test_model(self, trainer, config, testset, sampler):
        """The rules of testing models, depending on the specific models"""
        trainer.accuracy = trainer.super().test_model(config, testset, sampler)
