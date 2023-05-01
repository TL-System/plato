"""
The implemetation of the trainer for SMoG approach.
"""

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum

from plato.trainers import basic_ssl
from plato.trainers import loss_criterion



class Trainer(basic_ssl.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def get_loss_criterion(self):
        """Returns the loss criterion.
        As the loss functions derive from the lightly,
        it is desired to create a interface
        """

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
        self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)
        update_momentum(
            self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
        )
        update_momentum(
            self.model.projection_head,
            self.model.projection_head_momentum,
            m=self.momentum_val,
        )
