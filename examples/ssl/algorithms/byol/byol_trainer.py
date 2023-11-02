"""
A trainer for BYOL to rewrite the loss wrappe.
"""
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import update_momentum


from ssl import ssl_trainer
from plato.trainers import loss_criterion
from plato.config import Config


class Trainer(ssl_trainer.Trainer):
    """A trainer for BYOL to rewrite the loss wrappe."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def get_ssl_criterion(self):
        """A wrapper to connect ssl loss with plato."""
        defined_ssl_loss = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                loss = 0.5 * (
                    defined_ssl_loss(*outputs[0]) + defined_ssl_loss(*outputs[1])
                )
                return loss
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss

    def train_epoch_start(self, config):
        """
        Before the start of one epoch,
            prepare the momentum value for updating momentum outputs.
        """
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
                self.model.encoder, self.model.momentum_encoder, m=self.momentum_val
            )
            update_momentum(
                self.model.projector,
                self.model.momentum_projector,
                m=self.momentum_val,
            )
