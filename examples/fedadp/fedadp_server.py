"""
A federated learning training session using FedAdp.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""
from collections import OrderedDict

from plato.servers import fedavg

import logging
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from operator import itemgetter


class Server(fedavg.Server):
    """A federated learning server using the FedAdp algorithm."""
    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }
   
        return avg_update
