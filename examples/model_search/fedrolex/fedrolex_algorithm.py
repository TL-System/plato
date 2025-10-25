"""
FedRolexfl algorithm.
"""

from __future__ import annotations

import copy
import pickle
import random
import sys
from typing import Callable, cast

import ptflops
import torch
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the FedRolexfl algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate: float = 1.0
        self.model_class: Callable[..., Module] | None = None
        self.rates: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

    def _require_model(self) -> Module:
        model = self.model
        if model is None:
            raise RuntimeError("Model is not attached to the FedRolex algorithm.")
        return cast(Module, model)

    def _require_model_class(self) -> Callable[..., Module]:
        if self.model_class is None:
            raise RuntimeError("Model class has not been set via `choose_rate`.")
        return self.model_class

    def extract_weights(self, model: Module | None = None):
        model_to_use: Module
        if model is not None:
            model_to_use = model
        else:
            model_to_use = self._require_model()
        cpu_model = model_to_use.cpu()
        self.model = cpu_model
        return self.get_local_parameters()

    def choose_rate(
        self,
        limitation: tuple[float, float],
        model_class: Callable[..., Module],
    ) -> float:
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
            last = 0.0
            while True:
                rate = (smallest + biggest) / 2
                if abs(last - rate) < 0.01:
                    break
                pre_model = model_class(
                    model_rate=rate, **Config().parameters.client_model._asdict()
                )
                payload = pre_model.state_dict()
                size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
                in_channel = (
                    1 if hasattr(Config().parameters.client_model, "channels") else 3
                )
                macs, _ = ptflops.get_model_complexity_info(
                    pre_model,
                    (in_channel, 32, 32),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                if macs is None:
                    raise RuntimeError(
                        "Unable to compute model complexity for FedRolex."
                    )
                macs_value = float(macs) / 1024**2
                if macs_value <= limitation[1] and size <= limitation[0]:
                    smallest = rate
                else:
                    biggest = rate
                last = rate
            self.current_rate = rate
        else:
            rate = float(random.choice(self.rates))
            self.current_rate = rate
        return self.current_rate

    def get_local_parameters(self):
        """
        Get the parameters of local models from the global model.
        """
        model = self._require_model()
        model_class = self._require_model_class()
        current_rate = self.current_rate
        pre_model = model_class(
            model_rate=current_rate, **Config().parameters.client_model._asdict()
        )
        local_parameters = pre_model.state_dict()
        for key, value in model.state_dict().items():
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
        model = self._require_model()
        global_parameters = copy.deepcopy(model.state_dict())
        for key, value in model.state_dict().items():
            if "weight" in key or "bias" in key:
                count = torch.zeros(value.shape, device=value.device)
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
                        ] += torch.ones(local_weights[key].shape, device=value.device)
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
                        ] += torch.ones(local_weights[key].shape, device=value.device)
                    elif value.dim() == 2:
                        global_parameters[key][
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                        ] += torch.ones(local_weights[key].shape, device=value.device)
                    elif value.dim() == 1:
                        global_parameters[key][: local_weights[key].shape[0]] += (
                            copy.deepcopy(local_weights[key])
                        )
                        count[: local_weights[key].shape[0]] += torch.ones(
                            local_weights[key].shape, device=value.device
                        )
                count = torch.where(
                    count == 0, torch.ones(count.shape, device=value.device), count
                )
                global_parameters[key] = torch.div(
                    global_parameters[key] - value, count
                )
        return global_parameters

    # pylint:disable=too-many-branches
    def sort_channels(self):
        "Sort channels according to L2 norms."
        model = self._require_model()
        argindex = None
        shortcut_index_in = None
        parameters = model.state_dict()
        # pylint:disable=too-many-nested-blocks
        for key, value in parameters.items():
            # Sort the input channels according to the sequence of last output channels
            if argindex is not None:
                if "conv1" in key and key != "conv1.weight":
                    shortcut_index_in = copy.deepcopy(argindex)
                if value.dim() == 1:
                    if "linear" not in key and "mlp_head.1.bias" not in key:
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
            if (value.dim() == 4 and value.shape[1] > 1) or value.dim() == 2:
                if (
                    "linear" not in key
                    and "shortcut" not in key
                    and "to_patch_embedding" not in key
                    and "to_qkv" not in key
                ):
                    argindex = torch.arange(value.shape[0])
                    argindex = torch.roll(argindex, 1, -1)
                    parameters[key] = copy.deepcopy(parameters[key][argindex])
        model.load_state_dict(parameters)
