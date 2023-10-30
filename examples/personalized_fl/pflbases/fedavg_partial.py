"""
An algorithm for loading, aggregating, and extracting partial modules from a single model.

In some scenarios, given one defined model, the users want to utilize the sub-modules as 
the global model in federated learning. Thus, solely these desired sub-modules will be 
extracted and aggregated during the learning process. Thus, the algorithm proposes to 
support this feature by setting the hyper-parameter `global_module_names` in the config file.

The format of this hyper-parameter should be a list containing the names of the desired layers.

For example, when utilizing the "LeNet5" as the target model, the `global_module_names` can
be defined as:

    global_module_names:
        - conv1
        - conv2

By doing so, the conv1 and conv2 layers will be extracted.
"""

import string
from collections import OrderedDict
from typing import List, Optional

import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A base algorithm for extracting sub-modules from a model."""

    def get_target_weights(self, model_parameters: dict, module_names: List[str]):
        """Get target weights from model parameters based on the module name."""
        parameters_data = model_parameters.items()
        extracted_weights = OrderedDict(
            [
                (name, param)
                for name, param in parameters_data
                if any(
                    param_name in name.strip().split(".") for param_name in module_names
                )
            ]
        )
        return extracted_weights

    def extract_weights(
        self,
        model: Optional[torch.nn.Module] = None,
        module_names: Optional[List[str]] = None,
    ):
        """
        Extract weights from modules of the model.
        By default, weights of the entire model will be extracted.
        """
        model = self.model if model is None else model

        module_names = (
            module_names
            if module_names is not None
            else (
                Config().algorithm.global_module_names
                if hasattr(Config().algorithm, "global_module_names")
                else None
            )
        )

        # When the `global_module_names` is not set and
        # the `module_name` is not provided, this function
        # returns the whole model.
        if module_names is None:
            return model.cpu().state_dict()
        else:
            return self.get_target_weights(
                model.cpu().state_dict(), module_names=module_names
            )

    @staticmethod
    def extract_module_names(parameter_names):
        """Extract module names from given parameter names."""

        extracted_names = []
        # Remove punctuation and split the strings into words and sub-words
        translator = str.maketrans("", "", string.punctuation)
        combined_subnames = [
            [subname.translate(translator).lower() for subname in word.split(".")]
            for word in parameter_names
        ]

        # An indicator of the level where the strings begin to differ
        diff_level = 0
        # Find the point where the strings begin to differ in content
        for level, subnames in enumerate(zip(*combined_subnames)):
            if len(set(subnames)) > 1:
                diff_level = level
                break
        # Increase the level by 1
        diff_level += 1

        # Split the parameter names based on the `split_str`,
        # which should be a point as these names are presented
        # as 'encoder.xxx.xxx'
        # Extract the corresponding module names
        split_str = "."
        for para_name in parameter_names:
            splitted_names = para_name.split(split_str)
            core_names = splitted_names[:diff_level]
            module_name = f"{split_str}".join(core_names)
            if module_name not in extracted_names:
                extracted_names.append(module_name)

        return extracted_names

    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""

        self.model.load_state_dict(weights, strict=False)
