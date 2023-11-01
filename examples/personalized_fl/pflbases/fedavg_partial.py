"""
Loading the model weights for personalized FL clients.
"""
import os
import logging
from collections import OrderedDict

import torch
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
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            filename = f"{model_path}/{model_name}_{self.client_id}_local_layers.pth"
            if os.path.exists(filename):
                local_layers = torch.load(filename, map_location=torch.device("cpu"))
                # Remove after local layers are loaded.
                # Will save local layers after new weights are trained.
                os.remove(filename)

                weights.update(local_layers)

                logging.info(
                    "[Client #%d] Replaced portions of the global model with local layers.",
                    self.trainer.client_id,
                )

        self.model.load_state_dict(weights, strict=True)

    def extract_weights(self, model=None):
        weights = super().extract_weights(model)
        # Save the local layers before giving it to outbound processor.
        if hasattr(Config().algorithm, "local_layer_names"):
            # Extract weights of desired local layers
            local_layers = OrderedDict(
                [
                    (name, param)
                    for name, param in weights.items()
                    if any(
                        param_name in name.strip().split(".")
                        for param_name in Config().algorithm.local_layer_names
                    )
                ]
            )
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            filename = f"{model_path}/{model_name}_{self.client_id}_local_layers.pth"
            torch.save(local_layers, filename)
        return weights
