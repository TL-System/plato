"""
Implementation of our contrastive adaptation trainer.

"""

import os
import logging
import time
from attr import has

import numpy as np
import torch
import torch.nn.functional as F

from plato.config import Config
from plato.trainers import contrastive_ssl
from plato.utils import data_loaders_wrapper
from plato.utils import optimizers

from pFLCMA_losses import pFLCMALoss


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the BYOL client. """

    @staticmethod
    def loss_criterion(model):
        """ The loss computation. """
        temperature = Config().trainer.temperature
        base_temperature = Config().trainer.base_temperature
        contrast_mode = Config().trainer.contrast_mode
        batch_size = Config().trainer.batch_size
        contrastive_adaptation_criterion = pFLCMALoss(
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            batch_size=batch_size)

        return contrastive_adaptation_criterion
