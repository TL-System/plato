"""
The implementation of the contrastive adaptation's algorithm

"""

from collections import OrderedDict

from plato.algorithms import fedavg_ssl
from plato.config import Config
from plato.trainers.base import Trainer


class Algorithm(fedavg_ssl.Algorithm):
    """ Federated averaging algorithm for Byol models, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)

        # the most important parameter beta in the moving average
        # update method.
        # as the  Exponential Moving Average (EMA) update method is the most commonly used
        # method in the self-supervised learning method. This mechanism
        # should be supported as the foundation.
        self.default_moving_average_scale = 0.999

    def load_weights_moving_average(self, weights, average_scale=None):
        """ Loading the moel weights passed in as a parameter
            and assign to the target model based on the moving
            average method. """

        if hasattr(Config().trainer,
                   "model_ema_update") and Config().trainer.model_ema_update:

            if average_scale is None:
                average_scale = self.default_moving_average_scale

            model_ema_update = ModelEMA(average_scale)
            model_ema_update.update_model_moving_average(
                previous_model=self.model, current_model=weights)
        else:
            self.model.load_state_dict(weights, strict=False)
