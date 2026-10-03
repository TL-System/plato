"""
HeteroFL algorithm.
"""

from __future__ import annotations

import copy
import pickle
import random
import sys
from typing import Callable, cast

import numpy as np
import ptflops
import torch
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the HeteroFL algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate: float = 1.0
        self.model_class: Callable[..., Module] | None = None
        self.size_complexities = np.zeros(5, dtype=float)
        self.flops_complexities = np.zeros(5, dtype=float)
        self.rates: np.ndarray = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])

    def _require_model(self) -> Module:
        model = self.model
        if model is None:
            raise RuntimeError("Model is not attached to the HeteroFL algorithm.")
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
            for index, rate_value in enumerate(self.rates):
                rate = float(rate_value)
                if self.size_complexities[index] == 0:
                    pre_model = model_class(
                        model_rate=rate, **Config().parameters.client_model._asdict()
                    )
                    payload = pre_model.state_dict()
                    size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
                    self.size_complexities[index] = size
                    macs, _ = ptflops.get_model_complexity_info(
                        pre_model,
                        (3, 32, 32),
                        as_strings=False,
                        print_per_layer_stat=False,
                        verbose=False,
                    )
                    if macs is None:
                        raise RuntimeError("Unable to compute FLOPs for HeteroFL.")
                    macs_value = float(macs) / 1024**2
                    self.flops_complexities[index] = macs_value
                if index == self.rates.shape[0] - 1 or (
                    self.size_complexities[index] <= limitation[0]
                    and self.flops_complexities[index] <= limitation[1]
                ):
                    self.current_rate = rate
                    break
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
                    count == 0,
                    torch.ones(count.shape, device=value.device),
                    count,
                )
                global_parameters[key] = torch.div(
                    global_parameters[key] - value, count
                )
        return global_parameters

    def stat(self, model_class, trainloader):
        """
        The implementation of sBN.
        """
        model = self._require_model()
        with torch.no_grad():
            model_param = Config().parameters.model._asdict()
            if "track" in model_param:
                model_param.pop("track")
            test_model = model_class(track=True, **model_param)
            test_model.load_state_dict(model.state_dict(), strict=False)
            test_model.train(True)
            for example, _ in trainloader:
                test_model(example)
        return test_model
