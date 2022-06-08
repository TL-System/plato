"""
Implementation of our contrastive adaptation trainer.

"""

import torch

from plato.config import Config
from plato.trainers import contrastive_ssl

from contras_adap_losses import ContrasAdapLoss


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the FedProx client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation.

        """
        # define the loss computation instance
        defined_temperature = Config().trainer.temperature
        defined_contrast_mode = Config().trainer.contrast_mode
        base_temperature = Config().trainer.base_temperature
        criterion = ContrasAdapLoss(temperature=defined_temperature,
                                    contrast_mode=defined_contrast_mode,
                                    base_temperature=base_temperature)

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            encoded_z1, encoded_z2 = outputs
            loss = criterion(encoded_z1, encoded_z2)
            return loss

        return loss_compute