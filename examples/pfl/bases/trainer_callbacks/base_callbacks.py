"""
Callbacks for the personalized trainer.

The `PersonalizedLogProgressCallback` is provided as a default callback for
the personalized trainer.

The `PersonalizedModelCallback` is the base callback to 
    1). record the personalized model at every `model_logging_epoch_interval`
        of training.
    2). record he personalized model at the end of training.
 

The `PersonalizedMetricCallback` is the base callback to
    1). test the personalized model before the training.
    2). test the personalized model at every epoch.
    3). test the personalized model after the training.


The user is desired to inherit from these two basic callbacks to create
customize ones.
"""

import os
import logging

from plato.callbacks import trainer as trainer_callbacks
from plato.utils.filename_formatter import NameFormatter


class PersonalizedLogProgressCallback(trainer_callbacks.LogProgressCallback):
    """
    A trainer logging callback which controls the frequent of logging for
    both normal and personalized learning processes.
    """

    def on_train_step_end(self, trainer, config, batch=None, loss=None, **kwargs):
        log_iter_interval = (
            (
                config["logging_iteration_interval"]
                if config["logging_iteration_interval"] is not None
                else -1
            )
            if "logging_iteration_interval" in config
            else 10
        )

        if batch % log_iter_interval == 0:
            super().on_train_step_end(trainer, config, batch, loss, **kwargs)

    def on_train_epoch_end(self, trainer, config, **kwargs):
        log_epoch_interval = (
            config["logging_epoch_interval"]
            if "logging_epoch_interval" in config
            else 1
        )
        current_epoch = trainer.current_epoch

        if current_epoch % log_epoch_interval == 0:
            if trainer.client_id == 0:
                logging.info(
                    "[Server #%d] End of Epoch: [%d/%d][%d]\t Averaged Loss: %.6f",
                    os.getpid(),
                    trainer.current_epoch,
                    config["epochs"],
                    len(trainer.train_loader),
                    trainer.run_history.get_latest_metric("train_loss"),
                )
            else:
                logging.info(
                    "[Client #%d] End of Epoch: [%d/%d][%d]\t Averaged Loss: %.6f",
                    trainer.client_id,
                    trainer.current_epoch,
                    config["epochs"],
                    len(trainer.train_loader),
                    trainer.run_history.get_latest_metric("train_loss"),
                )


class PersonalizedMetricCallback(trainer_callbacks.TrainerCallback):
    """A trainer callback to compute and record the test accuracy of the
    personalized model."""

    def on_train_run_start(self, trainer, config, **kwargs):
        super().on_train_run_start(trainer, config, **kwargs)
        # perform test for the personalized model
        trainer.perform_personalized_metric_checkpoint(config)

    def on_train_epoch_end(self, trainer, config, **kwargs):
        super().on_train_epoch_end(trainer, config, **kwargs)
        # perform test for the personalized model
        trainer.perform_personalized_metric_checkpoint(config)

    def on_train_run_end(self, trainer, config, **kwargs):
        super().on_train_run_end(trainer, config, **kwargs)
        # perform test for the personalized model
        trainer.perform_personalized_metric_checkpoint(config)


class PersonalizedModelCallback(trainer_callbacks.TrainerCallback):
    """A trainer callback to record the personalized model
    every `model_logging_epoch_interval` epochs and at the end of
    training."""

    def on_train_epoch_end(self, trainer, config, **kwargs):
        super().on_train_epoch_end(trainer, config, **kwargs)
        log_epoch_interval = (
            config["model_logging_epoch_interval"]
            if "model_logging_epoch_interval" in config
            else 1
        )
        current_epoch = trainer.current_epoch

        if current_epoch % log_epoch_interval == 0:
            if "max_concurrency" in config:
                trainer.perform_personalized_model_checkpoint(config, current_epoch)

    def on_train_run_end(self, trainer, config, **kwargs):
        super().on_train_run_end(trainer, config, **kwargs)

        if "max_concurrency" in config:
            trainer.perform_personalized_model_checkpoint(config, **kwargs)
