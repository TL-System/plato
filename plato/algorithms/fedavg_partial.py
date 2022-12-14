"""
The enhanced federated averaging algorithm to aggregate partial sub-modules of one model.

Utilization condition:
    In some scenarios, even given one defined model, the users want to utilize the partial
    sub-modules as the global model in federated learning. Thus, solely these desired
    sub-modules will be extracted and aggregated during the learning process.
    Then, noticing that the names of parameters in one sub-module hold consistent names,
    we propose this piece of code to support the aforementioned feature by setting the
    hyper-parameter 'global_submodules_name' in the config file.

The format of this hyper-parameter should be:
{submodule1_prefix}__{submodule2_prefix}__{submodule3_prefix}__...

names for different submodules are separated by two consecutive underscores.


For example
 A. Given the defined whole model: encoder + head, the hyper-parameter in the config
    file 'global_submodules_name' under the 'trainer' can be set to:
    - whole     : utilizing the whole model as the global model
    - encoder   : utilizing the encoder as the global model
    - head      : utilizing the head as the global model

    Demo:

        trainer:
            global_submodules_name: encoder


 B. Given the defined whole model: encoder1 + encoder2 + encoder3 + head1 + head2,
    the hyper-parameter in the config file 'global_submodules_name' under the 'trainer'
    can be set to
    - whole                 : utilizing the whole model as the global model
    - encoder1__encoder2    : utilizing solely encoder1 and encoder2 as the global model
    - encoder2__head1       : utilizing solely encoder2 and head as the global model
    - encoder2__head1__head2: utilizing solely encoder2, head1 and head2 as the global model

    Demo:

        trainer:
            global_submodules_name: encoder1__encoder2

"""

import logging
from typing import List, Optional
from collections import OrderedDict

import torch

from plato.algorithms import fedavg
from plato.config import Config
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):
    """Federated averaging algorithm for the partial aggregation, used by both the client and the server."""

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)

        # in this algorithm, the sub-module's name that is used as the global model
        # shared among clients should be set.
        # by default, the whole model will be used as the global model
        # i.e., whole_model_name = "whole"
        self.whole_model_name = "whole"

        # the separator used to combine different names into one
        # string.
        # by default, two consecutive underscores are utilized.
        self.separator = "__"

    def extract_weights(
        self,
        model: Optional[torch.nn.Module] = None,
        submodules_name: Optional[List[str]] = None,
    ):
        """Extract weights from submodules of the model.
        By default, weights of the whole model will be extracted."""

        submodules_name = (
            submodules_name
            if submodules_name is not None
            else (
                Config().trainer.global_submodules_name.split(self.separator)
                if hasattr(Config().trainer, "global_submodules_name")
                else self.whole_model_name.split(self.separator)
            )
        )

        logging.info("Extracting parameters with names %s.", submodules_name)

        model = self.model if model is None else model

        if self.whole_model_name in submodules_name:
            return model.cpu().state_dict()

        full_weights = model.cpu().state_dict()
        extracted_weights = OrderedDict(
            [
                (name, param)
                for name, param in full_weights.items()
                if any([param_name in name for param_name in submodules_name])
            ]
        )
        return extracted_weights

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
    def extract_submodules_name(parameters_name):
        """Extracting submodules name from given parameters' names."""

        extracted_names = []

        # the para name is presented as encoder.xxx.xxx
        # that is combined by the key_word "."
        # we aim to extract the encoder
        split_str = "."
        for para_name in parameters_name:
            splitted_names = para_name.split(split_str)
            core_name = splitted_names[0]
            # add the obtained prefix to the list, if
            #   - empty list
            #   - a new prefix
            # add to the empty list directly
            if core_name not in extracted_names:
                extracted_names.append(core_name)

        return extracted_names
