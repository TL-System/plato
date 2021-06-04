"""
A federated learning client using FedNova, and the local number of epochs is randomly
generated and communicated to the server at each communication round.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

import logging
import random
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple
from plato.clients import base

@dataclass
class Report(base.Report):
    """A client report containing the number of local epochs."""
    epochs: int


class Client(simple.Client):
    """A fednova federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.pattern = None
        self.max_local_iter = None

    def configure(self):
        super().configure()
        random.seed(3000 + self.client_id)

    async def train(self):
        """ FedNova clients use different number of local epochs. """

        # generate the number of local epochs randomly
        if Config().algorithm.pattern == "constant":
            local_epochs = Config().algorithm.max_local_epochs
        else:
            local_epochs = random.randint(2,
                                          Config().algorithm.max_local_epochs)

        logging.info("[Client #%d] Training with %d epoches.", self.client_id,
                     local_epochs)

        # Perform model training for a specific number of epoches
        Config().trainer = Config().trainer._replace(epochs=local_epochs)

        report, weights = await super().train()

        return Report(report.num_samples, report.accuracy, local_epochs), weights
