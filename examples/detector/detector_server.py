"""
A customized server with detector so that poisoned model updates can be filtered out.
"""

import logging

from plato.config import Config
from plato.servers import fedavg
from collections import OrderedDict

import numpy as np
import torch

class Server(fedavg.Server):

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
    def configure():
