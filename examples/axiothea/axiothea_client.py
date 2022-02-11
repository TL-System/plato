"""
A federated learning client using Axiothea.

A client clips and adds Gaussian noise to its model gradients,
quantizes the weights, and sends them as its update to its edge server.

"""
import logging
import time

from plato.clients import simple


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
        comm_time = time.time()

        return simple.Report(report.num_samples, report.accuracy,
                             report.training_time, comm_time,
                             report.update_response), weights
