"""
A federated learning client for AFL.

Reference:

Goetz et al., "Active Federated Learning".

https://arxiv.org/pdf/1909.12641.pdf
"""

import logging
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple

import math


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

        return Report(report.num_samples, report.accuracy, report.training_time,
                      report.data_loading_time, valuation), weights
    
    def get_loss(self):
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.loss"
        loss = self.trainer.load_loss(filename)
        return loss

    def calc_valuation(self, num_samples, loss):
        """Calculate the valuation value based on #samples and loss value."""
        valuation = float(1 / math.sqrt(num_samples)) * loss 
        return valuation
