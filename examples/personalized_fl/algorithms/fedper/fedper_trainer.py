"""
A personalized federated learning trainer for FedPer.
"""

from pflbases import personalized_trainer
from pflbases import trainer_utils

from plato.config import Config


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """According to FedPer,
        1. freeze body of the model during personalization.
        """
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_module_names,
            )

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.activate_model(
                self.model, Config().algorithm.global_module_names
            )
