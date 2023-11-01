"""
Loading the model weights for personalized FL clients.
"""
import os
import logging
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm for extracting layers from a model."""

    def load_weights(self, weights):
        """
        Loads the first part of the model with the global received from the server
            and the second part of the model with the saved local model.
        """
        if hasattr(Config().algorithm, "local_layer_names"):
            # Load the local model that has previously been saved
            filename = f"client_{self.trainer.client_id}_local_model.pth"
            location = Config().params["checkpoint_path"]
            if os.path.exists(os.path.join(location, filename)):
                local_layer_names = Config().algorithm.local_layer_names

                # Extract weights of desired local layers
                local_layers = OrderedDict(
                    [
                        (name, param)
                        for name, param in enumerate(self.model.cpu().state_dict())
                        if any(
                            param_name in name.strip().split(".")
                            for param_name in local_layer_names
                        )
                    ]
                )

                weights.update(local_layers)

                logging.info(
                    "[Client #%d] Replaced portions of the global model with local layers.",
                    self.trainer.client_id,
                )

        self.model.load_state_dict(weights, strict=True)
