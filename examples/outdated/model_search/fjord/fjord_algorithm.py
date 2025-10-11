"""
FjORD algorithm.
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
    """A federated learning algorithm using the FjORD algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate = 1
        self.model_class = None
        self.size_complexities = np.zeros(5)
        self.flops_complexities = np.zeros(5)
        self.rates = np.array([1, 0.5, 0.25, 0.125, 0.0625])

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
            for index, rate in enumerate(self.rates):
                if self.size_complexities[index] == 0:
                    pre_model = model_class(
                        model_rate=rate, **Config().parameters.client_model._asdict()
                    )
                    payload = pre_model.state_dict()
                    size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
                    self.size_complexities[index] = size
                    if hasattr(Config().parameters.model, "channels"):
                        channel_width = Config().parameters.model.channels
                    else:
                        channel_width = 3
                    macs, _ = ptflops.get_model_complexity_info(
                        pre_model,
                        (channel_width, 32, 32),
                        as_strings=False,
                        print_per_layer_stat=False,
                        verbose=False,
                    )
                    macs /= 1024**2
                    self.flops_complexities[index] = macs
                if index == self.rates.shape[0] - 1 or (
                    self.size_complexities[index] <= limitation[0]
                    and self.size_complexities[index] <= limitation[1]
                ):
                    self.current_rate = rate
                    break
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
                for local_weights in weights_received:
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
