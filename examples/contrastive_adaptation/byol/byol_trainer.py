"""
Implementation of the Byol's trainer.

"""

import torch.nn.functional as F

from plato.trainers import contrastive_ssl


def mean_squared_error(x, y):
    """ Compute the mean square error. """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def loss_fn_with_stop_gradients(outputs):
    """ Compute the errors with stop gradients mechanism. """
    (online_pred_one, online_pred_two), (target_proj_one,
                                         target_proj_two) = outputs

    # use the detach mechanism to stop the gradient for target learner
    loss_one = mean_squared_error(online_pred_one, target_proj_two.detach())
    loss_two = mean_squared_error(online_pred_two, target_proj_one.detach())

    loss = loss_one + loss_two
    return loss.mean()


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the BYOL client. """

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
