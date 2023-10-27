"""
Implementation of the trainer for FedEMA.

This trainer is the one used in BYOL method of FedSSL.
"""

from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import update_momentum

from plato.trainers import loss_criterion
from pflbases import ssl_trainer


class Trainer(ssl_trainer.Trainer):
    """A trainer for BYOL to rewrite the loss wrappe."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def plato_ssl_loss_wrapper(self):
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
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        global_epoch = (self.current_round - 1) * config["epochs"] + epoch
        if not self.do_final_personalization:
            self.momentum_val = cosine_schedule(global_epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)
        if not self.do_final_personalization:
            update_momentum(
                self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
            )
            update_momentum(
                self.model.projection_head,
                self.model.projection_head_momentum,
                m=self.momentum_val,
            )
