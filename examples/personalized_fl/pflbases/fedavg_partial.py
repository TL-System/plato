"""
An algorithm for extracting partial modules from a model.

These modules can be set by the `global_module_names` hyper-parameter in the 
configuration file.

For example, with the LeNet-5 model, `global_module_names` can be defined as:

    global_module_names:
        - conv1
        - conv2
"""

from collections import OrderedDict
from typing import List, Optional

import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm for extracting modules from a model."""

    def extract_weights(
        self,
        model: Optional[torch.nn.Module] = None,
        module_names: Optional[List[str]] = None,
    ):
        """Extract weights from the model based on the given module names."""
        model = self.model if model is None else model

        if module_names is None and hasattr(Config().algorithm, "global_module_names"):
            module_names = Config().algorithm.global_module_names

        if module_names is None:
            # When `global_module_names` is not set and `module_name` is not provided,
            # return all the model weights
            return model.cpu().state_dict()
        else:
            return Algorithm.get_module_weights(
                model.cpu().state_dict(), module_names=module_names
            )

    def load_weights(self, weights):
        """Loads a portion of the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)

    @staticmethod
    def get_module_weights(model_parameters: dict, module_names: List[str]):
        """Get weights from model parameters based on module names."""
        return OrderedDict(
            [
                (name, param)
                for name, param in model_parameters.items()
                if any(
                    param_name in name.strip().split(".") for param_name in module_names
                )
            ]
        )
