"""
Implement the server for SimCLR method.

"""

import logging
from plato.servers import fedavg


class Server(fedavg.Server):

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
