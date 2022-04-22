"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""
import logging
import math
import time
from dataclasses import dataclass

from plato.clients import simple
from plato.config import Config


@dataclass
class Report(simple.Report):
    """A client report containing the valuation, to be sent to the AFL federated learning server."""
    valuation: float


class Client(simple.Client):
    """A federated learning client for AFL."""
    async def train(self):
        logging.info("Training on AFL client #%d", self.client_id)

        report, weights = await super().train()

        # Get the valuation indicating the likely utility of training on this client
        loss = self.get_loss()
        valuation = self.calc_valuation(report.num_samples, loss)
        comm_time = time.time()

        return Report(report.num_samples, report.accuracy,
                      report.training_time, comm_time, report.update_response,
                      valuation), weights

    def get_loss(self):
        """ Retrieve the loss value from the training process. """
        model_name = Config().trainer.model_name
        filename = f'{model_name}_{self.client_id}.loss'
        loss = self.trainer.load_loss(filename)
        return loss

    def calc_valuation(self, num_samples, loss):
        """ Calculate the valuation value based on the number of samples and loss value. """
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
