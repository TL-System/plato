"""
Implement the trainer for Fedrep method.

"""

import torch
import numpy as np

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)

        self.model_representation_weights_key = []
        self.model_head_weights_key = []

    def set_global_local_weights_key(self, global_keys):
        """ Setting the global local weights key. """
        # the representation keys are obtained from
        #   the server response
        self.model_representation_weights_key = global_keys
        # the left weights are regarded as the head in default
        full_model_weights_key = self.model.state_dict().keys()
        self.model_head_weights_key = filter(
            lambda i: i not in self.model_representation_weights_key,
            full_model_weights_key)
