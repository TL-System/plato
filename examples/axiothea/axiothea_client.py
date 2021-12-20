"""
A federated learning client using Axiothea.

A client clips and adds Gaussian noise to its model gradients,
quantizes the weights, and sends them as its update to its edge server.

"""
from dataclasses import dataclass
import logging

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """Report from an Axiothea client, to be sent to the server."""


class Client(simple.Client):
    """
    A federated learning client with support for the Axiothea Algorithm which
    adds noise to the gradients and quantizes new weights on the client side.
    """
    async def train(self):
        logging.info("[Client #%d] Training on an Axiothea client.",
                     self.client_id)

        # Perform model training
        report, weights = await super().train()

        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.data_loading_time), weights
