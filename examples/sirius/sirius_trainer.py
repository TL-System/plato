"""
A federated learning trainer used by Sirius client.
"""

import logging
import os

from plato.config import Config
from plato.trainers import basic
import numpy as np

from plato.trainers import  tracking

from inspect import currentframe


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class LossTracker(tracking.LossTracker):
    def __init__(self):
        super().__init__()
        self.loss_decay = 1e-2

    def reset(self):

        """do not reset this loss tracker."""

    def update(self, loss_batch_value, batch_size=1):
        """Updates the loss tracker with another loss value from a batch."""
        self.total_loss = (1. - self.loss_decay) * self.total_loss \
                                                   + self.loss_decay * loss_batch_value
        
    @property
    def average(self):
        """"Recording for each epoch"""
        """But we only need the last epoch's for each local training"""
        return np.sqrt(self.total_loss.cpu().detach().item())

class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)
        self._loss_tracker = LossTracker()

