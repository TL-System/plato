"""
An algorithm for extracting partial layers from a model.

These layers can be set by the `global_layer_names` hyper-parameter in the 
configuration file.

For example, with the LeNet-5 model, `global_layer_names` can be defined as:

    global_layer_names:
        - conv1
        - conv2
"""

from collections import OrderedDict
from typing import List, Optional

import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm for extracting layers from a model."""

    def load_weights(self, weights):
        """Loads a portion of the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)

    def extract_local_weights(self, local_layer_names: List[str]):
        """Get weights from model parameters based on layer names."""
        return OrderedDict(
            [
                (name, param)
                for name, param in enumerate(self.model.cpu().state_dict())
                if any(
                    param_name in name.strip().split(".")
                    for param_name in local_layer_names
                )
            ]
        )
