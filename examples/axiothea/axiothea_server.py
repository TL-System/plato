"""
A cross-silo federated learning server using Axiothea, as either edge or central servers.
"""

from plato.config import Config
from plato.utils import quantizer
from plato.utils import dp_gaussian

from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using Axiothea."""
    async def aggregate_weights(self, updates):
        """Dequantize clients' weights and then aggregates them."""
        if hasattr(Config().algorithm,
                   'quantization') and not Config().algorithm.quantization:
            update = await self.federated_averaging(updates)
        else:
            dequantized_updates = self.dequantize_client_updates(updates)
            update = await self.federated_averaging(dequantized_updates)

        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

    def dequantize_client_updates(self, reports):
        """Dequantize clients' updated weights."""
        dequantized_updates = []
        for (report, payload) in reports:
            dequantized_weights = quantizer.dequantize_model_weights(payload)
            dequantized_updates.append((report, dequantized_weights))

        return dequantized_updates

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to its clients. """
        if Config().is_edge_server():
            noise_added_weights = dp_gaussian.gaussian_mechanism(
                payload, 'edge_server_downlink')
        else:
            noise_added_weights = dp_gaussian.gaussian_mechanism(
                payload, 'central_downlink')

        if hasattr(Config().algorithm,
                   'quantization') and not Config().algorithm.quantization:
            return noise_added_weights

        quantized_weights = quantizer.quantize_model_weights(
            noise_added_weights)
        return quantized_weights
