"""
Implementation of the SimSiam's trainer.

"""

import torch.nn.functional as F

from plato.trainers import contrastive_ssl


def negative_cosine_similarity(x, y):
    """ Compute the negative cosine similarity. """
    # x = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(y, dim=-1, p=2)
    # return -(x * y).sum(dim=1).mean()
    return -F.cosine_similarity(x, y.detach(), dim=-1).mean()


def loss_fn_with_stop_gradients(outputs):
    """ Compute the errors with stop gradients mechanism. """
    (encoded_h1, encoded_h2), (predicted_z1, predicted_z2) = outputs

    # use the detach mechanism to stop the gradient for target learner
    loss_one = negative_cosine_similarity(predicted_z1, encoded_h2.detach())
    loss_two = negative_cosine_similarity(predicted_z2, encoded_h1.detach())

    loss = loss_one / 2 + loss_two / 2
    return loss


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the SimSiam client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation.
        """

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            loss = loss_fn_with_stop_gradients(outputs)
            return loss

        return loss_compute
