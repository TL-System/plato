"""
Implement the trainer for base siamese method.

"""
import os
import time
import logging

import torch
import numpy as np

from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):

    def __init__(self, model=None):
        super().__init__(model)

        self.model_representation_weights_key = []
        self.model_head_weights_key = []
