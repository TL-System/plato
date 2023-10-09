"""
A personalized federated learning trainer for FedBABU.
"""


from pflbases import personalized_trainer
from pflbases import trainer_utils

from plato.config import Config


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """According to FedBABU,
        1. freeze head of the model during federated training phase.
        2. freeze body of the personalized model during personalized learning phase.
        """
        super().train_run_start(config)
        if self.personalized_learning:
            trainer_utils.freeze_model(
                self.personalized_model,
                Config().algorithm.global_modules_name,
                log_info=f"[Client #{self.client_id}]",
            )
        else:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.personalized_modules_name,
                log_info=f"[Client #{self.client_id}]",
            )

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        if self.personalized_learning:
            trainer_utils.activate_model(
                self.personalized_model, Config().algorithm.global_modules_name
            )
        else:
            trainer_utils.activate_model(
                self.model, Config().algorithm.personalized_modules_name
            )
