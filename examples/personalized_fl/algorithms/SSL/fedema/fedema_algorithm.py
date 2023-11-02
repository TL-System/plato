"""
A algorithm used by FedEMA approach.
"""
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """Extract the encoder of the model."""

    def extract_encoder(self):
        """Extract the encoder."""
        encoder_layer_names = Config().algorithm.encoder_layer_names
        return OrderedDict(
            [
                (name, param)
                for name, param in self.model.state_dict().items()
                if any(
                    param_name in name.strip().split(".")
                    for param_name in encoder_layer_names
                )
            ]
        )
