"""
A federated learning client using FedNova, and the local number of epochs is randomly
generated and communicated to the server at each communication round.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

from dataclasses import dataclass
import logging
import numpy as np

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
    def configure(self):
        super().configure()
        np.random.seed(3000 + self.client_id)

    async def train(self):
        """ FedNova clients use different number of local epochs. """

        # generate the number of local epochs randomly
        if hasattr(
                Config().algorithm,
                'pattern') and Config().algorithm.pattern == "uniform_random":
            local_epochs = np.random.randint(
                2,
                Config().algorithm.max_local_epochs + 1)
            # Perform model training for a specific number of epoches
            Config().trainer = Config().trainer._replace(epochs=local_epochs)

            logging.info("[Client #%d] Training with %d epoches.",
                         self.client_id, local_epochs)

        report, weights = await super().train()

        return Report(report.num_samples, report.accuracy,
                      report.training_time,
                      Config().trainer.epochs), weights
