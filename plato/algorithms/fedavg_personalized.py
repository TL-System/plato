"""
A personalized federate learning algorithm that loads and saves local layers of a model.
"""

import logging
import os
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config
from plato.serialization.safetensor import deserialize_tree, serialize_tree


class Algorithm(fedavg.Algorithm):
    """
    A personalized federate learning algorithm that loads and saves local layers
    of a model.
    """

    def load_weights(self, weights):
        """
        Loads local layers included in `local_layer_names` to the received weights which
        will be loaded to the model
        """
        if hasattr(Config().algorithm, "local_layer_names"):
            # Get the filename of the previous saved local layer
            model_path = Config().params["model_path"]
            model_name = Config().trainer.model_name
            filename = (
                f"{model_path}/{model_name}_{self.client_id}_local_layers.safetensors"
            )

            # Load local layers to the weights when the file exists

            if os.path.exists(filename):
                with open(filename, "rb") as local_file:
                    serialized = local_file.read()
                payload = deserialize_tree(serialized)
                local_layers = OrderedDict(payload.items())
            else:
                local_layers = None

            if local_layers:
                # Update the received weights with the loaded local layers
                weights.update(local_layers)

                logging.info(
                    "[Client #%d] Replaced portions of the global model with local layers.",
                    self.trainer.client_id,
                )

        self.model.load_state_dict(weights, strict=True)

    def save_local_layers(self, local_layers, filename):
        """
        Save local layers to a file with the filename provided.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not filename.endswith(".safetensors"):
            raise ValueError(
                f"Personalized layer checkpoints must end with '.safetensors': {filename}"
            )

        serialized = serialize_tree(local_layers)
        with open(filename, "wb") as local_file:
            local_file.write(serialized)
