"""
A personalized federated learning trainer with FedPer.
"""
from plato.config import Config
from plato.trainers import basic
from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """
    A trainer with FedPer, which freezes the global model layers in the final
    personalization round.
    """

    def train_run_start(self, config):
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
