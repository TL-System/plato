"""
A personalized federated learning trainer for FedPer.
"""
from plato.config import Config
from plato.trainers import basic
from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """A trainer to freeze and activate layers of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """According to FedPer,
        Freeze body of the model during personalization.
        """
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_layer_names,
            )

    def train_run_end(self, config):
        """Activate the model."""
        super().train_run_end(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.activate_model(
                self.model, Config().algorithm.global_layer_names
            )
