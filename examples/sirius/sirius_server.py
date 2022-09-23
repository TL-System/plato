"""
An asynchronous federated learning server using Sirius.
"""

import torch

from plato.config import Config
from plato.servers import fedavg

class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)