"""
A customized trainer for FEI
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from afl import afl_trainer
from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from plato.models import registry as models_registry


class Trainer(afl_trainer.Trainer):
    """A custom trainer for FEI. """
