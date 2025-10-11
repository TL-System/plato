"""
AnyCostfl algorithm.
"""

import copy
import pickle
import random
import sys

import numpy as np
import ptflops
import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the AnyCostfl algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate = 1
        self.model_class = None
        self.rates = [1.0, 0.5, 0.25, 0.125, 0.0625]

    def extract_weights(self, model=None):
        self.model = self.model.cpu()
        payload = self.get_local_parameters()
        return payload

    def choose_rate(self, limitation, model_class):
        """
        Choose a compression rate based on current limitation.
        Update the sub model for the client.
        """
        self.model_class = model_class
        if (
            hasattr(Config().parameters, "limitation")
            and hasattr(Config().parameters.limitation, "activated")
            and Config().parameters.limitation.activated
        ):
            smallest = 0.5
            biggest = 1.0
            last = 0
            while True:
                rate = (smallest + biggest) / 2
                if (abs(last - rate)) < 0.01:
                    break
                pre_model = model_class(
                    model_rate=rate, **Config().parameters.client_model._asdict()
                )
                payload = pre_model.state_dict()
                size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
                if hasattr(Config().parameters.client_model, "channels"):
                    in_channel = 1
                else:
                    in_channel = 3
                macs, _ = ptflops.get_model_complexity_info(
                    pre_model,
                    (in_channel, 32, 32),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                macs /= 1024**2
                if macs <= limitation[1] and size <= limitation[0]:
                    smallest = rate
                else:
                    biggest = rate
                last = rate
            self.current_rate = rate
        else:
            # In the original implementation, the rate are uniformly sampled
            rate = random.choice(self.rates)
            self.current_rate = rate
        return self.current_rate

    def get_local_parameters(self):
        """
        Get the parameters of local models from the global model.
        """
        current_rate = self.current_rate
        pre_model = self.model_class(
            model_rate=current_rate, **Config().parameters.client_model._asdict()
        )
        local_parameters = pre_model.state_dict()
        for key, value in self.model.state_dict().items():
            if "weight" in key or "bias" in key:
                if value.dim() == 4 or value.dim() == 2:
                    local_parameters[key] = copy.deepcopy(
                        value[
                            : local_parameters[key].shape[0],
                            : local_parameters[key].shape[1],
                            ...,
                        ]
                    )
                elif value.dim() == 3:
                    local_parameters[key] = copy.deepcopy(
                        value[
                            : local_parameters[key].shape[0],
                            : local_parameters[key].shape[1],
                            : local_parameters[key].shape[2],
                            ...,
                        ]
                    )
                else:
                    local_parameters[key] = copy.deepcopy(
                        value[: local_parameters[key].shape[0]]
                    )
        return local_parameters

    def aggregation(self, weights_received):
        """
        Aggregate weights of different complexities.
        """
        global_parameters = copy.deepcopy(self.model.state_dict())
        for key, value in self.model.state_dict().items():
            if "weight" in key or "bias" in key:
                count = torch.zeros(value.shape)
                for _, local_weights in enumerate(weights_received):
                    if value.dim() == 4:
                        global_parameters[key][
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            ...,
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            ...,
                        ] += torch.ones(local_weights[key].shape)
                    if value.dim() == 3:
                        global_parameters[key][
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            : local_weights[key].shape[2],
                            ...,
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            : local_weights[key].shape[2],
                            ...,
                        ] += torch.ones(local_weights[key].shape)
                    elif value.dim() == 2:
                        global_parameters[key][
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                        ] += torch.ones(local_weights[key].shape)
                    elif value.dim() == 1:
                        global_parameters[key][: local_weights[key].shape[0]] += (
                            copy.deepcopy(local_weights[key])
                        )
                        count[: local_weights[key].shape[0]] += torch.ones(
                            local_weights[key].shape
                        )
                count = torch.where(count == 0, torch.ones(count.shape), count)
                global_parameters[key] = torch.div(
                    global_parameters[key] - value, count
                )
        return global_parameters

    # pylint:disable=too-many-branches
    def sort_channels(self):
        "Sort channels according to L2 norms."
        argindex = None
        shortcut_index_in = None
        parameters = self.model.state_dict()
        # pylint:disable=too-many-nested-blocks
        for key, value in parameters.items():
            # Sort the input channels according to the sequence of last output channels
            if argindex is not None:
                if "conv1" in key and not key == "conv1.weight":
                    shortcut_index_in = copy.deepcopy(argindex)
                if value.dim() == 1:
                    if not "linear" in key and not "mlp_head.1.bias" in key:
                        parameters[key] = copy.deepcopy(value[argindex])
                elif value.dim() > 1:
                    if "shortcut" in key:
                        parameters[key] = copy.deepcopy(
                            value[argindex, ...][:, shortcut_index_in, ...]
                        )
                    else:
                        if not ("to_out" in key and "weight" in key):
                            if value.dim() == 4 and value.shape[1] == 1:
                                parameters[key] = copy.deepcopy(value[argindex, ...])
                            else:
                                parameters[key] = copy.deepcopy(value[:, argindex, ...])
                    # If this is a conv or linear, we need to sort the channels.
            if (value.dim() == 4 and value.shape[1] > 1) or value.dim() == 2:
                if (
                    not "linear" in key
                    and not "shortcut" in key
                    and not "to_patch_embedding" in key
                    and not "to_qkv" in key
                ):
                    dims = (1, 2, 3) if value.dim() == 4 else (1)
                    l2_norm = torch.norm(value, p=2, dim=dims)
                    argindex = torch.argsort(l2_norm, descending=True)
                    parameters[key] = copy.deepcopy(parameters[key][argindex])
        self.model.load_state_dict(parameters)
