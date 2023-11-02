"""
A base trainer for MoCoV2 algorithm.
"""


from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from ssl import ssl_trainer
from plato.config import Config


class Trainer(ssl_trainer.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        global_epoch = (self.current_round - 1) * config["epochs"] + epoch
        if not self.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(global_epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """
        At the start of every iteration,
            update the models for generating momentum
            with new momemtum parameter: momentum value.
        """
        super().train_step_start(config)
        if not self.current_round > Config().trainer.rounds:
            update_momentum(
                self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
            )
            update_momentum(
                self.model.projector,
                self.model.projector_momentum,
                m=self.momentum_val,
            )
