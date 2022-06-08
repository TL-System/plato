"""
A personalized federated learning server using BYOL.

Reference:

Grill et al., "Bootstrap your own latent: A new approach to self-supervised Learning".

https://arxiv.org/pdf/2006.07733.pdf

Source code: https://github.com/lucidrains/byol-pytorch
"""

import logging
from collections import OrderedDict

from plato.servers import fedavg


class Server(fedavg.Server):
    """A personalized federated learning server using the BYOL algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

    def compute_weight_deltas(self, updates):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, __, payload, __) in updates]
        deltas = []
        for weight in weights_received:
            delta = list()
            for name, _ in weight.items():
                delta.append(name)

            deltas.append(delta)

        return self.algorithm.compute_weight_deltas(weights_received)
