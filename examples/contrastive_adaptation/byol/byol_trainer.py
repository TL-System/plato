"""
Implementation of the Byol's trainer.

"""

from plato.trainers import contrastive_ssl
from plato.utils import ssl_losses


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the BYOL client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation.
        """
        criterion = ssl_losses.CrossStopGradientL2loss()

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            loss = criterion(outputs)
            return loss

        return loss_compute
