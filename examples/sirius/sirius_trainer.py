"""
A federated learning trainer used by Sirius client.
"""

import logging
import os
import torch

from plato.config import Config
from plato.trainers import basic

import scaffold_optimizer


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)
