"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass

import logging

from plato.config import Config
from plato.clients import edge
from plato.utils import quantizer
from plato.utils import dp_gaussian


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

        # Apply Gaussian mechanism
        noise_added_weights = dp_gaussian.gaussian_mechanism(
            weights, 'edge_server_uplink')

        if hasattr(Config().algorithm,
                   'quantization') and not Config().algorithm.quantization:
            return Report(report.client_id, report.num_samples,
                          report.accuracy, report.training_time,
                          0), noise_added_weights

        # Quantize weights before sending
        quantized_weights = quantizer.quantize_model_weights(
            noise_added_weights)

        return Report(report.client_id, report.num_samples, report.accuracy,
                      report.training_time, 0), quantized_weights
