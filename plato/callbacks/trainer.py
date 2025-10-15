"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new trainer callbacks.

Defines a default callback to print training progress.
"""

import logging
import os
from abc import ABC

from plato.utils import fonts


def resolve_num_samples(trainer):
    """Best-effort estimate of the number of samples available for training."""
    sampler = getattr(trainer, "sampler", None)

    if sampler is not None:
        if hasattr(sampler, "num_samples"):
            try:
                return sampler.num_samples()
            except TypeError:
                pass
        if hasattr(sampler, "__len__"):
            try:
                return len(sampler)
            except TypeError:
                pass

    train_loader = getattr(trainer, "train_loader", None)
    if train_loader is not None:
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "__len__"):
            try:
                return len(dataset)
            except TypeError:
                pass

    trainset = getattr(trainer, "trainset", None)
    if trainset is not None and hasattr(trainset, "__len__"):
        try:
            return len(trainset)
        except TypeError:
            pass

    return None


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

    def on_test_outputs(self, trainer, outputs, **kwargs):
        """
        Event called to process model outputs during testing.

        :param outputs: the raw model outputs
        :return: processed outputs (default: unchanged)
        """
        return outputs


class LogProgressCallback(TrainerCallback):
    """
    A callback which prints a message at the start of each epoch, and at the end of each step.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """
        Event called at the start of training run.
        """
        num_samples = resolve_num_samples(trainer)
        if num_samples is not None:
            message = f"Loading the dataset with size {num_samples}."
        else:
            message = "Loading the dataset."

        if trainer.client_id == 0:
            logging.info(
                "[Server #%s] %s",
                os.getpid(),
                message,
            )
        else:
            logging.info(
                "[Client #%d] %s",
                trainer.client_id,
                message,
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
