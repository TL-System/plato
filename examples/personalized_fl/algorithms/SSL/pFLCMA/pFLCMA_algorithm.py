"""
The implementation of the contrastive adaptation's algorithm

"""
import copy
from collections import OrderedDict

from plato.algorithms import fedavg_pers
from plato.config import Config
from plato.trainers.base import Trainer

from moving_average import ModelEMA


class Algorithm(fedavg_pers.Algorithm):
    """ Federated averaging algorithm for Byol models, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)

    def load_weights_moving_average(self, weights, average_scale=None):
        """ Loading the moel weights passed in as a parameter
            and assign to the target model based on the moving
            average method. """

        model_ema_update = ModelEMA(average_scale)

        existed_model_weights = self.extract_weights()

        moving_averaged_weights = model_ema_update.update_parameters_moving_average(
            previous_parameters=existed_model_weights,
            current_parameters=weights)

        self.load_weights(moving_averaged_weights)
