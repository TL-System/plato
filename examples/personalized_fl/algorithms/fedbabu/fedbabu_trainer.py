"""
A personalized federated learning trainer for FedBABU.
"""
from plato.config import Config
from plato.trainers import basic
from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """A trainer to freeze and activate layers of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """According to FedBABU,
        1. freeze first part of the model during federated training phase.
        2. freeze second part of the personalized model during personalized learning phase.
        """
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_layer_names,
            )
        else:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.local_layer_names,
            )

    def train_run_end(self, config):
        """Activate the model."""
        super().train_run_end(config)

        if self.current_round > Config().trainer.rounds:
            trainer_utils.activate_model(
                self.model, Config().algorithm.global_layer_names
            )
        else:
            trainer_utils.activate_model(
                self.model, Config().algorithm.local_layer_names
            )
