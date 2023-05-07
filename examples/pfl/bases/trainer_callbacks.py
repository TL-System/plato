"""
Callbacks for the personalized trainer.
"""

import os
import logging

from plato.callbacks import trainer as trainer_callbacks
from plato.config import Config

from bases.trainer_utils import checkpoint_personalized_accuracy


class PersonalizedLogMetricCallback(trainer_callbacks.TrainerCallback):
    """A trainer callback to compute and record the test accuracy of the
    personalized model."""

    def on_train_run_start(self, trainer, config, **kwargs):
        super().on_train_run_start(trainer, config, **kwargs)
        # perform test for the personalized model
        result_path = Config().params["result_path"]
        if trainer.personalized_learning:
            test_outputs = trainer.test_personalized_model(config)

            checkpoint_personalized_accuracy(
                result_path,
                client_id=trainer.client_id,
                accuracy=test_outputs["accuracy"],
                current_round=trainer.current_round,
                epoch=trainer.current_epoch,
                run_id=None,
            )

    def on_train_epoch_end(self, trainer, config, **kwargs):
        # perform test for the personalized model
        self.on_train_run_start(trainer, config, **kwargs)


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
