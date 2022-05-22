"""
An honest-but-curious federated learning server with gradient leakage attack.

Reference:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
"""

import math

import numpy as np
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ An honest-but-curious federated learning server with gradient leakage attack. """

    def __init__(self):
        super().__init__()
