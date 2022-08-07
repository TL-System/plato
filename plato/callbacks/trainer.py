"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new trainer callbacks.

Defines a default callback to print training progress.
"""
import logging
import os
from abc import ABC

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


class PrintProgressCallback(TrainerCallback):
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
        Method called at the end of a training step.

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
