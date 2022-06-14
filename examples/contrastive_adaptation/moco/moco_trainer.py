"""
Implementation of MoCo's trainer

"""

from torch import nn

from plato.trainers import contrastive_ssl


def loss_fn_cn(outputs):
    """ Compute the loss. """
    logits, labels = outputs
    criterion = nn.CrossEntropyLoss()

    return criterion(logits, labels)


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the MOCO client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation.
        """

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            loss = loss_fn_cn(outputs)
            return loss

        return loss_compute
