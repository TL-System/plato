"""
The trainer for the fedavg algorithm with fine-tuning in personalization round.
"""
from pflbases import trainer_utils

from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """A trainer to create a personalized training process after the final round."""

    def train_run_start(self, config):
        """Freeze the body during personalization."""
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            # Freeze the model body while only optimizing the head
            # during the final personalization
            trainer_utils.freeze_model(
                self.model, Config().algorithm.global_layer_names
            )
            # Set the number of epochs for personalization
            config["epochs"] = Config().algorithm.personalization.epochs
