"""
A self-supervised federated learning trainer with MoCoV2.
"""


from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.trainers import self_supervised_learning as ssl_trainer
from plato.config import Config


class Trainer(ssl_trainer.Trainer):
    """
    A trainer with MoCoV2, which updates the momentum value at the start
    of each training epoch and updates the model based on this value in a
    momentum manner in each training step.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # The momentum value used to update the model
        # with Exponential Moving Average
        self.momentum_val = 0

    def train_epoch_start(self, config):
        """Update the momentum value."""
        super().train_epoch_start(config)
        total_epochs = config["epochs"] * config["rounds"]
        global_epoch = (self.current_round - 1) * config["epochs"] + self.current_epoch
        # Compute the momentum value during the regular federated training process
        if not self.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(global_epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """
        At the start of every iteration, the model should be updated based on the
        momentum value.
        """
        super().train_step_start(config)
        if not self.current_round > Config().trainer.rounds:
            # Update the model based on the momentum value
            # Specifically, it updates parameters of `encoder` with
            # Exponential Moving Average of `encoder_momentum`
            update_momentum(
                self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
            )
            update_momentum(
                self.model.projector,
                self.model.projector_momentum,
                m=self.momentum_val,
            )
