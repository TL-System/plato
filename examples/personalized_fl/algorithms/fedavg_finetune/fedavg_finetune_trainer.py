"""
The trainer for the fedavg algorithm with fine-tuning.
"""


from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """A trainer to create a personalized training process after the final round."""

    def train_run_start(self, config):
        """A new train model to load the epochs for personalization."""
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            config["epochs"] = Config().algorithm.personalization.epochs
