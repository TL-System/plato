"""
A personalized federate learning algorithm that loads and saves local layers of a model.
"""

import logging
import os
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config
from plato.serialization.safetensor import deserialize_tree, serialize_tree
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path


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
        trainer = self.require_trainer()
        if hasattr(Config().algorithm, "local_layer_names"):
            # Get the filename of the previous saved local layer
            model_path = Config().params["model_path"]
            model_name = Config().trainer.model_name
            filename = checkpoint_path(
                model_path, checkpoint_name(
                    model_name, self.client_id, "local_layers", suffix=".safetensors"
                )
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
                    trainer.client_id,
                )

        model = self.require_model()
        model.load_state_dict(weights, strict=True)

    def save_local_layers(self, local_layers, filename):
        """
        Save local layers within the configured model root.

        The legacy client supplies an assembled logical-name path. Recognize
        that exact default and encode it here, keeping naming authority with
        the matching reader. Other explicit paths retain their spelling.
        """
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        legacy_default = (
            f"{model_path}/{model_name}_{self.client_id}_local_layers.safetensors"
        )
        joined_default = os.path.join(
            model_path, f"{model_name}_{self.client_id}_local_layers.safetensors"
        )
        if os.fspath(filename) in {legacy_default, joined_default}:
            relative = checkpoint_name(
                model_name, self.client_id, "local_layers", suffix=".safetensors"
            )
        else:
            relative = (
                os.path.relpath(filename, model_path)
                if os.path.isabs(filename) else filename
            )
        filename = checkpoint_path(model_path, relative)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not filename.endswith(".safetensors"):
            raise ValueError(
                f"Personalized layer checkpoints must end with '.safetensors': {filename}"
            )

        serialized = serialize_tree(local_layers)
        with open(filename, "wb") as local_file:
            local_file.write(serialized)
