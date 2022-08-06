"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new trainer callbacks. 

Defines two default callbacks to log metrics and print training progress.
"""
import os
import logging
from abc import ABC


def get_default_callbacks():
    """
    Obtains a list of default trainer callbacks.
    """
    default_callbacks = PrintProgressCallback

    return default_callbacks


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new trainer callbacks.
    """

    def on_init_end(self, trainer, **kwargs):
        """
        Event called at the end of trainer initialisation.
        """

    def on_training_run_start(self, trainer, **kwargs):
        """
        Event called at the start of training run.
        """

    def on_train_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """

    def on_train_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """

    def on_train_step_end(self, trainer, batch, loss, **kwargs):
        """
        Event called at the end of a training step.

        :param batch: the current batch of training data.
        :param loss: the loss computed in the current batch.
        """

    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of a training epoch.
        """

    def on_eval_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of an evaluation epoch.
        """

    def on_eval_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of an evaluation step.

        :param batch: the current batch of evaluation data.
        :param batch_output: the outputs returned by
            :meth:`plato.trainers.base.Trainer.calculate_eval_batch_loss`.
        """

    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of evaluation.
        """

    def on_training_run_epoch_end(self, trainer, **kwargs):
        """
        Event called during a training run after both training and evaluation epochs have
        been completed.
        """

    def on_training_run_end(self, trainer, **kwargs):
        """
        Event called at the end of training run.
        """

    def on_evaluation_run_start(self, trainer, **kwargs):
        """
        Event called at the start of an evaluation run.
        """

    def on_evaluation_run_end(self, trainer, **kwargs):
        """
        Event called at the end of an evaluation run.
        """

    def on_stop_training_error(self, trainer, **kwargs):
        """
        Event called when a stop training error is raised.
        """


class PrintProgressCallback(TrainerCallback):
    """
    A callback which prints a message at the start and end of a run,
    as well as at the start of each epoch.
    """

    def on_training_run_start(self, trainer, **kwargs):
        logging.info("\nStarting training run")

    def on_train_epoch_start(self, trainer, **kwargs):
        logging.info(f"\nStarting epoch {trainer.current_epoch}")

    def on_train_step_end(self, trainer, batch, loss, **kwargs):
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
                    trainer.total_epochs,
                    batch,
                    len(trainer.train_loader),
                    loss.data.item(),
                )
            else:
                logging.info(
                    "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                    trainer.client_id,
                    trainer.current_epoch,
                    trainer.total_epochs,
                    batch,
                    len(trainer.train_loader),
                    loss.data.item(),
                )

    def on_evaluation_run_start(self, trainer, **kwargs):
        logging.info("\nStarting evaluation run")

    def on_evaluation_run_end(self, trainer, **kwargs):
        logging.info("Finishing evaluation run")
