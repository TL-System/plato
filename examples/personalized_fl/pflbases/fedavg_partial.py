"""
An algorithm for extracting partial modules from a model.

These modules can be set by the `global_module_names` hyper-parameter in the 
configuration file.

For example, with the LeNet-5 model, `global_module_names` can be defined as:

    global_module_names:
        - conv1
        - conv2
"""

import string
from collections import OrderedDict
from typing import List, Optional

import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm for extracting modules from a model."""
    def get_module_weights(self, model_parameters: dict, module_names: List[str]):
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
            return self.get_module_weights(
                model.cpu().state_dict(), module_names=module_names
            )

    @staticmethod
    def extract_module_names(parameter_names):
        """Extract module names from given parameter names."""
        # The split string to split the parameter name
        # Generally, the parameter name is a list of sub-names
        # connected by '.',
        # such as encoder.conv1.weight.
        split_str = "."

        # Remove punctuation and
        # split parameter name into sub-names
        # such as from encoder.conv1.weight to [encoder, conv1, weight].
        translator = str.maketrans("", "", string.punctuation)
        splitted_names = [
            [subname.translate(translator).lower() for subname in name.split(split_str)]
            for name in parameter_names
        ]

        # A position idx where the sub-names of different parameters
        # begin to differ
        # for example, with [encoder, conv1, weight], [encoder, conv1, bais]
        # the diff_idx will be 1.
        diff_idx = 0
        for idx, subnames in enumerate(zip(*splitted_names)):
            if len(set(subnames)) > 1:
                diff_idx = idx
                break

        # Extract the first `diff_idx` parameter names
        # as module names.
        extracted_names = []
        for para_name in parameter_names:
            splitted_names = para_name.split(split_str)
            core_names = splitted_names[: diff_idx + 1]
            module_name = f"{split_str}".join(core_names)
            if module_name not in extracted_names:
                extracted_names.append(module_name)

        return extracted_names

    def load_weights(self, weights):
        """Loads a portion of the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)
