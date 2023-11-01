"""
An algorithm for extracting partial layers from a model.

These layers can be set by the `global_layer_names` hyper-parameter in the 
configuration file.

For example, with the LeNet-5 model, `global_layer_names` can be defined as:

    global_layer_names:
        - conv1
        - conv2
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
        # First, load the global model weights.
        super().load_weights(weights)

        # Second update the current model weights with local model weights saved.
        # Not load local weights if there is no saved local model.
        if hasattr(Config().algorithm, "local_layer_names"):
            # Load the local model weights previously saved on filesystem.
            filename = f"client_{self.trainer.client_id}_local_model.pth"
            location = Config().params["checkpoint_path"]
            if os.path.exists(os.path.join(location, filename)):
                local_layer_names = Config().algorithm.local_layer_names
                self.trainer.load_model(filename, location=location)

                # Extract weights of desired local layers
                local_model_weights = OrderedDict(
                    [
                        (name, param)
                        for name, param in self.model.state_dict().items()
                        if any(
                            param_name in name.strip().split(".")
                            for param_name in local_layer_names
                        )
                    ]
                )
                weights.update(local_model_weights)

                logging.info(
                    "[Client #%d] Replaced portions of the global model with local layers.",
                    self.trainer.client_id,
                )

                # Load the weights containing two parts of the model weights.
                self.model.load_state_dict(weights, strict=True)
