"""
The enhanced federated averaging algorithm to aggregate partial sub-modules of one model.

Utilization condition:
    In some scenarios, even given one defined model, the users want to utilize the partial
    sub-modules as the global model in federated learning. Thus, solely these desired
    sub-modules will be extracted and aggregated during the learning process.
    Then, noticing that the names of parameters in one sub-module hold consistent names,
    we propose this piece of code to support the aforementioned feature by setting the
    hyper-parameter `global_modules_name` in the config file.

The format of this hyper-parameter should be a list containing the name of desired layers.


For example, when utilizing the "LeNet5" as the target model, the `global_modules_name` can
be defined as:

    global_modules_name:
        - conv1
        - conv2

thus, the conv1 and conv2 layers will be used as the global model.
"""

import os
import string
import logging
from typing import List, Optional
from collections import OrderedDict

import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """Federated averaging algorithm for the partial aggregation, used by both the client and the server."""

    def get_algorithm_holder(self):
        """Get who holds the defined algorithm."""
        return (
            f"server #{os.getpid()}"
            if self.client_id == 0
            else f"Client #{self.client_id}"
        )

    def extract_weights(
        self,
        model: Optional[torch.nn.Module] = None,
        modules_name: Optional[List[str]] = None,
    ):
        """Extract weights from modules of the model.
        By default, weights of the whole model will be extracted."""
        model = self.model if model is None else model
        modules_name = (
            modules_name
            if modules_name is not None
            else (
                Config().trainer.global_modules_name
                if hasattr(Config().trainer, "global_modules_name")
                else None
            )
        )
        # when no modules are required,
        # return the whole model
        if modules_name is None:
            return model.cpu().state_dict()
        else:

            logging.info(
                "[%s] Extracting parameters with names %s.",
                self.get_algorithm_holder(),
                modules_name,
            )

            return OrderedDict(
                [
                    (name, param)
                    for name, param in model.cpu().state_dict().items()
                    if any([param_name in name.strip().split(".") for param_name in modules_name])
                ]
            )

    def is_consistent_weights(self, weights_param_name):
        """Whether the 'weights' holds the parameters' name the same as the self.model."""

        model_params_name = self.model.state_dict().keys()

        search_func = lambda x, y: [x_i for x_i in x if x_i not in y]

        inconsistent_params = []
        if len(model_params_name) > len(weights_param_name):
            inconsistent_params = search_func(model_params_name, weights_param_name)
        else:
            inconsistent_params = search_func(weights_param_name, model_params_name)

        return len(inconsistent_params) == 0, inconsistent_params

    def load_weights(
        self,
        weights: dict,
        strict: bool = False,
    ):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=strict)

    @staticmethod
    def extract_modules_name(parameters_name):
        """Extracting modules name from given parameters' names."""

        extracted_names = []
        # Remove punctuation and split the strings into words and sub-words
        translator = str.maketrans("", "", string.punctuation)
        combined_subnames = [
            [subname.translate(translator).lower() for subname in word.split(".")]
            for word in parameters_name
        ]

        # from which subname, the modules name show difference
        diff_level = 0
        # Find the point where the strings begin to differ in content
        for level, subnames in enumerate(zip(*combined_subnames)):
            if len(set(subnames)) > 1:
                diff_level = level
                break
        # add 1 to begin from 0
        diff_level += 1
        # the para name is presented as encoder.xxx.xxx
        # that is combined by the key_word "."
        # we aim to extract the encoder
        split_str = "."
        for para_name in parameters_name:
            splitted_names = para_name.split(split_str)
            core_names = splitted_names[:diff_level]
            module_name = f"{split_str}".join(core_names)
            if module_name not in extracted_names:
                extracted_names.append(module_name)

        return extracted_names
