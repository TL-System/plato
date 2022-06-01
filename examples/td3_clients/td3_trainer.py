"""
A customized trainer for td3.
"""
import logging
import os
import time

import numpy as np
import torch
from opacus.privacy_engine import PrivacyEngine
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers

class Trainer(basic.Trainer):
    def __init__(self):
        print("hello")






