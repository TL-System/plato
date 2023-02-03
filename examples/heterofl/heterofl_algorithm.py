"""
HeteroFL algorithm.
"""

import sys
import pickle
import random
import copy

from collections import OrderedDict
import torch
import ptflops
import numpy as np
from plato.config import Config
from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the Hermes algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate = 1
        self.current_subnet = None
        self.size_complexities = np.zeros(5)
        self.flops_complexities = np.zeros(5)
        self.rates = np.array([1, 0.5, 0.25, 0.125, 0.0625])

    def extract_weights(self, model=None):
        payload = self.current_subnet.cpu().state_dict()
        return payload

    def choose_rate(self, limitation, model_class):
        """
        Choose a compression rate based on current limitation.
        Update the sub model for the client.
        """
        # for index, rate in enumerate(self.rates):
        #     if self.size_complexities[index] == 0:
        #         pre_model = model_class(
        #             rate, **Config().parameters.cllient_model._asdict()
        #         )
        #         payload = pre_model.state_dict()
        #         size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
        #         self.size_complexities[index] = size
        #         macs, _ = ptflops.get_model_complexity_info(
        #             pre_model,
        #             (3, 32, 32),
        #             as_strings=False,
        #             print_per_layer_stat=False,
        #             verbose=False,
        #         )
        #         macs /= 1024**2
        #         self.flops_complexities[index] = macs
        #     if index == self.rates.shape[0] - 1 or (
        #         self.size_complexities[index] <= limitation[0]
        #         and self.size_complexities[index] <= limitation[1]
        #     ):
        #         self.current_rate = rate
        #         self.current_subnet = model_class(
        #             rate, **Config().parameters.client_model._asdict()
        #         )
        #         break

        # In the original implementation, the rate are uniformly sampled

        rate = random.choice(self.rates)
        self.current_rate = rate
        self.current_subnet = model_class(
            rate, **Config().parameters.client_model._asdict()
        )

        local_parameters = self.get_local_parameters()
        self.current_subnet.load_state_dict(local_parameters)
        return self.current_rate

    def get_local_parameters(self):
        """
        Get the parameters of local models from the global model.
        """
        current_rate = self.current_rate
        local_parameters = OrderedDict()
        for key, value in self.model.state_dict().items():
            if "weight" in key or "bias" in key:
                if value.dim() == 4:
                    if key == "conv1.weight":
                        local_parameters[key] = copy.deepcopy(
                            value[: int(current_rate * value.shape[0]), :]
                        )
                    else:
                        local_parameters[key] = copy.deepcopy(
                            value[
                                : int(current_rate * value.shape[0]),
                                : int(current_rate * value.shape[1]),
                                :,
                            ]
                        )
                elif value.dim() == 2:
                    local_parameters[key] = copy.deepcopy(
                        value[
                            :,
                            : int(current_rate * value.shape[1]),
                        ]
                    )
                elif value.dim() == 1:
                    if not "linear" in key:
                        local_parameters[key] = copy.deepcopy(
                            value[: int(current_rate * value.shape[0])]
                        )
                    else:
                        local_parameters[key] = copy.deepcopy(value)
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
                        if key == "conv1.weight":
                            global_parameters[key][
                                : local_weights[key].shape[0], :
                            ] += copy.deepcopy(local_weights[key])
                            count[: local_weights[key].shape[0], :] += torch.ones(
                                local_weights[key].shape
                            )
                        else:
                            global_parameters[key][
                                : local_weights[key].shape[0],
                                : local_weights[key].shape[1],
                                :,
                            ] += copy.deepcopy(local_weights[key])
                            count[
                                : local_weights[key].shape[0],
                                : local_weights[key].shape[1],
                                :,
                            ] += torch.ones(local_weights[key].shape)
                    elif value.dim() == 2:
                        global_parameters[key][
                            :,
                            : local_weights[key].shape[1],
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            :,
                            : local_weights[key].shape[1],
                        ] += torch.ones(local_weights[key].shape)
                    elif value.dim() == 1:
                        if not "linear" in key:
                            global_parameters[key][
                                : local_weights[key].shape[0]
                            ] += copy.deepcopy(local_weights[key])
                            count[: local_weights[key].shape[0]] += torch.ones(
                                local_weights[key].shape
                            )
                        else:
                            global_parameters[key] += copy.deepcopy(local_weights[key])
                            count += torch.ones(local_weights[key].shape)
                count = torch.where(count == 0, torch.ones(count.shape), count)
                global_parameters[key] = torch.div(
                    global_parameters[key] - value, count
                )
        return global_parameters
