"""
A self-supervised federated learning trainer with SimSiam.
"""

from plato.trainers import loss_criterion
from plato.trainers import self_supervised_learning as ssl_trainer


class Trainer(ssl_trainer.Trainer):
    """A trainer with SimSiam to compute the loss."""

    def get_ssl_criterion(self):
        """Get the loss proposed by the SimSiam."""
        defined_ssl_loss = loss_criterion.get()

        def compute_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                loss = 0.5 * (
                    defined_ssl_loss(*outputs[0]) + defined_ssl_loss(*outputs[1])
                )
                return loss
            else:
                return defined_ssl_loss(outputs)

        return compute_loss
