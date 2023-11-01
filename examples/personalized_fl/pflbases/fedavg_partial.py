"""
A base algorithm to load and save local layers of a model.
"""
import os
import logging
from collections import OrderedDict

import torch
from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm to load local layers to the received weights and save
    local layers after the local training."""

    def load_weights(self, weights):
        """
        Loads local layers included in `local_layer_names` to the received weights which,
        will be loaded to the model.
        """
        if hasattr(Config().algorithm, "local_layer_names"):
            # Get the filename of the previous saved local layer
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            filename = f"{model_path}/{model_name}_{self.client_id}_local_layers.pth"
            # Load local layers to the weights when the file exists
            if os.path.exists(filename):
                local_layers = torch.load(filename, map_location=torch.device("cpu"))
                # Update the received weights with the loaded local layers
                weights.update(local_layers)

                logging.info(
                    "[Client #%d] Replaced portions of the global model with local layers.",
                    self.trainer.client_id,
                )

        self.model.load_state_dict(weights, strict=True)

    def extract_weights(self, model=None):
        weights = super().extract_weights(model)
        # Save local layers before giving them to the outbound processor
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
