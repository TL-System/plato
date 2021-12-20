"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass
import logging

from plato.clients import edge


@dataclass
class Report(edge.Report):
    """Report from an Axiothea edge server, to be sent to the central server."""


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    async def train(self):
        logging.info("[Edge Server #%d] Training on an Axiothea edge server.",
                     self.client_id)

        # Perform model training
        report, weights = await super().train()

        # Might apply Gaussian mechanism

        return Report(report.client_id, report.num_samples, report.accuracy,
                      report.average_accuracy, 0,
                      report.data_loading_time), weights
