"""
A base trainer for simsiam algorithm.
"""

from plato.trainers import loss_criterion

from pflbases import ssl_trainer


class Trainer(ssl_trainer.Trainer):
    """A trainer for SimSiam to rewrite the loss wrapper."""

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
