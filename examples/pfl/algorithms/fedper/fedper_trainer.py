"""
A personalized federated learning trainer for FedPer.

"""

from pflbases import personalized_trainer
from pflbases.trainer_utils import freeze_model, activate_model

from plato.config import Config


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """According to FedPer,
        1. freeze body of the model during personalization.
        """
        super().train_run_start(config)
        if self.personalized_learning:
            freeze_model(
                self.personalized_model,
                Config().algorithm.global_modules_name,
                log_info=f"[Client #{self.client_id}]",
            )

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        if self.personalized_learning:
            activate_model(
                self.personalized_model, Config().algorithm.global_modules_name
            )
