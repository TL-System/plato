"""
A federated learning client using Axiothea.

A client clips and adds Gaussian noise to its model weights,
quantizes the weights, and sends them as its update to its edge server.

"""
from dataclasses import dataclass
import logging

from plato.config import Config
from plato.clients import simple
from plato.utils import quantizer
from plato.utils import dp_gaussian


@dataclass
class Report(simple.Report):
    """Report from an Axiothea client, to be sent to the server."""


class Client(simple.Client):
    """
    A federated learning client with support for the Axiothea Algorithm which
    adds noise to the gradients and quantizes new weights on the client side.
    """
    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        if hasattr(Config().algorithm,
                   'quantization') and not Config().algorithm.quantization:
            super().load_payload(server_payload)
        else:
            dequantized_weights = quantizer.dequantize_model_weights(
                server_payload)
            self.algorithm.load_weights(dequantized_weights)

    async def train(self):
        logging.info("[Client #%d] Training on an Axiothea client.",
                     self.client_id)

        # Perform model training
        report, weights = await super().train()

        # Apply Gaussian mechanism
        noise_added_weights = dp_gaussian.gaussian_mechanism(
            weights, 'client_uplink')

        if hasattr(Config().algorithm,
                   'quantization') and not Config().algorithm.quantization:
            return Report(report.num_samples, report.accuracy,
                          report.training_time,
                          report.data_loading_time), noise_added_weights

        # Quantize weights before sending
        quantized_weights = quantizer.quantize_model_weights(
            noise_added_weights)

        return Report(report.num_samples, report.accuracy,
                      report.training_time,
                      report.data_loading_time), quantized_weights
