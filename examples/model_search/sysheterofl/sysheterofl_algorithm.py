"""
The algorithm for paper system-heterogenous federated learning through architecture search.
"""

from __future__ import annotations

import copy
import pickle
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

import ptflops
import torch
from resnet import ResnetWrapper
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config

SysHeteroConfig = tuple[list[int], list[float]]


# pylint:disable=too-many-instance-attributes
class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the ElasticArch algorithm."""

    def _require_model(self) -> ResnetWrapper:
        model = self.model
        if model is None:
            raise RuntimeError(
                "Model is not attached to the SysHeteroFL algorithm instance."
            )
        return cast(ResnetWrapper, model)

    def _require_model_class(self) -> Callable[..., ResnetWrapper]:
        if self.model_class is None:
            raise RuntimeError(
                "Model class has not been initialised; call `initialize_arch_map` first."
            )
        return self.model_class

    def _require_current_config(self) -> SysHeteroConfig:
        if self.current_config is None:
            raise RuntimeError("Current configuration has not been selected.")
        return self.current_config

    @staticmethod
    def _normalise_config(
        config: Sequence[Sequence[int | float]],
    ) -> SysHeteroConfig:
        if len(config) != 2:
            raise ValueError("SysHeteroFL expects configurations with two components.")
        depth_raw, width_raw = config
        depth = [int(value) for value in depth_raw]
        width = [float(value) for value in width_raw]
        return depth, width

    def __init__(self, trainer=None):
        super().__init__(trainer)
        limitation_params = Config().parameters.limitation
        self.current_config: SysHeteroConfig | None = None
        self.model_class: Callable[..., ResnetWrapper] | None = None
        self.epsilon: float = limitation_params.epsilon  # 0.8
        self.max_loop: int = limitation_params.max_loop  # 50
        self.configs: list[int] = []
        self.arch_list: Dict[int, SysHeteroConfig] = {}
        self.min_configs_size_flops: tuple[float, float] | None = None
        self.size_flops_counts_dict: list[list[float | int]] = []
        self.arch_counts: int = 0
        self.min_configs: list[SysHeteroConfig] = []
        self.biggest_net: SysHeteroConfig | None = None

    def extract_weights(self, model: Module | None = None):
        model_to_use: ResnetWrapper
        if model is not None:
            model_to_use = cast(ResnetWrapper, model)
        else:
            model_to_use = self._require_model()
        cpu_model = model_to_use.cpu()
        self.model = cpu_model
        return self.get_local_parameters()

    def initialize_arch_map(self, model_class: Callable[..., ResnetWrapper]):
        """
        First add largest net into the map.
        """
        self.model_class = model_class
        model = self._require_model()
        if Config().parameters.supernet.width and Config().parameters.supernet.depth:
            object.__setattr__(
                model,
                "channel_rate_lists",
                [[1.0], [1.0], [1.0], [1.0]],
            )
        for func in [min, max]:
            raw_config = model.get_net(func)
            config = self._normalise_config(raw_config)
            size, macs = self.calculate_flops_size(config)
            self.add_into_config(config, size, macs)
        if Config().parameters.supernet.width and Config().parameters.supernet.depth:
            object.__setattr__(
                model,
                "channel_rate_lists",
                [[0.5, 1.0], [0.5, 1.0], [1.0], [1.0]],
            )

    def add_into_config(self, config: SysHeteroConfig, size: float, macs: float):
        """
        Add the configuration into current arch database.
        """
        self.arch_list[self.arch_counts] = config
        self.size_flops_counts_dict.append([size, macs, 0])
        self.sort_config(size, macs)
        self.arch_counts += 1

    def sort_config(self, size: float, flops: float) -> None:
        """
        Sort configurations descending.
        """
        pos = 0
        for pos, config_index in enumerate(self.configs):
            if size > self.size_flops_counts_dict[config_index][0] or (
                size == self.size_flops_counts_dict[config_index][0]
                and flops >= self.size_flops_counts_dict[config_index][1]
            ):
                break
        self.configs.insert(pos, self.arch_counts)

    def calculate_flops_size(self, config: SysHeteroConfig):
        """
        Calculate the size and flops.
        """
        model_factory = self._require_model_class()
        depth, width = config
        pre_model = model_factory(
            configs=(depth, width), **Config().parameters.client_model._asdict()
        )
        size = sys.getsizeof(pickle.dumps(pre_model.state_dict())) / 1024**2
        macs, _ = ptflops.get_model_complexity_info(
            pre_model,
            (3, 32, 32),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        if macs is None:
            raise RuntimeError("Unable to compute model complexity statistics.")
        macs_value = float(macs) / 1024**2
        size = int(size * 10) / 10.0
        macs_value = int(macs_value * 10) / 10.0
        return size, macs_value

    def get_local_parameters(self):
        """
        Get the parameters of local models from the global model.
        """
        current_config = self._require_current_config()
        model_factory = self._require_model_class()
        model = self._require_model()
        pre_model = model_factory(
            configs=current_config, **Config().parameters.client_model._asdict()
        )
        local_parameters = pre_model.state_dict()

        for key, value in model.state_dict().items():
            if key in local_parameters:
                if value.dim() == 4:
                    local_parameters[key].data = copy.deepcopy(
                        value.data[
                            : local_parameters[key].shape[0],
                            : local_parameters[key].shape[1],
                            : local_parameters[key].shape[2],
                            : local_parameters[key].shape[3],
                        ]
                    )
                elif value.dim() == 2:
                    local_parameters[key].data = copy.deepcopy(
                        value.data[
                            : local_parameters[key].shape[0],
                            : local_parameters[key].shape[1],
                        ]
                    )
                elif value.dim() == 1:
                    local_parameters[key].data = copy.deepcopy(
                        value.data[: local_parameters[key].shape[0]]
                    )
        pre_model = model_factory(
            configs=current_config, **Config().parameters.client_model._asdict()
        )
        pre_model.load_state_dict(local_parameters)
        local_parameters = pre_model.state_dict()
        return local_parameters

    def aggregation(self, weights_received, update_track=True):
        """
        Aggregate weights of different complexities.
        """
        model = self._require_model()
        global_parameters = copy.deepcopy(model.state_dict())
        for key, value in model.state_dict().items():
            value_org = copy.deepcopy(value)
            count = torch.zeros(value.shape, device=value.device)
            for local_weights in weights_received:
                if key in local_weights:
                    if value.dim() == 4:
                        global_parameters[key][
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            : local_weights[key].shape[2],
                            : local_weights[key].shape[3],
                        ] += copy.deepcopy(local_weights[key])
                        count[
                            : local_weights[key].shape[0],
                            : local_weights[key].shape[1],
                            : local_weights[key].shape[2],
                            : local_weights[key].shape[3],
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
                        if update_track or (
                            not update_track
                            and not ("running" in key or "tracked" in key)
                        ):
                            global_parameters[key][: local_weights[key].shape[0]] += (
                                copy.deepcopy(local_weights[key])
                            )
                            count[: local_weights[key].shape[0]] += torch.ones(
                                local_weights[key].shape, device=value.device
                            )
            count = torch.where(count == 0, torch.ones(count.shape), count)
            global_parameters[key] = torch.div(
                global_parameters[key] - value_org, count
            )
        return global_parameters

    # pylint:disable=too-many-locals
    def choose_config(self, limitation: Sequence[float]):
        """
        Choose a compression rate based on current limitation.
        Update the sub model for the client.
        """
        size_limitation = int(limitation[0] * 10) / 10.0
        flops_limitation = int(limitation[1] * 10) / 10.0
        # Greedy to exploration
        current_config = self.find_arch(size_limitation, flops_limitation)
        count_loop = 0
        while current_config is None or random.random() > self.epsilon:
            if count_loop > self.max_loop:
                raise RuntimeError(
                    "Cannot find suitable model in current search space!"
                )
            new_arch, size, macs = self.find_new_arch()
            if not new_arch in self.arch_list.values():
                self.add_into_config(new_arch, size, macs)
            current_config = self.find_arch(size_limitation, flops_limitation)
            count_loop += 1
        if current_config is None:
            raise RuntimeError(
                "Unable to select a configuration for the provided limitations."
            )
        chosen_config = self.arch_list[current_config]
        self.current_config = chosen_config
        self.size_flops_counts_dict[current_config][2] += 1
        self.update_biggest(self.current_config)
        return self.current_config

    def update_biggest(self, new_config: SysHeteroConfig):
        "Update the configuration of the biggest net."
        if self.biggest_net is None:
            depth, width = new_config
            self.biggest_net = ([*depth], [*width])
        else:
            depth, width = self.biggest_net
            new_depth, new_width = new_config
            for index, new in enumerate(new_depth):
                depth[index] = max(depth[index], new)
            for index, new in enumerate(new_width):
                width[index] = max(width[index], new)

    def find_arch(self, size_limitation, flops_limitation):
        """
        Find the arch satisfying the requirement.
        """
        for config_index in self.configs:
            if (
                self.size_flops_counts_dict[config_index][0] <= size_limitation
                and self.size_flops_counts_dict[config_index][1] <= flops_limitation
            ):
                return config_index
        return None

    def find_new_arch(self):
        """
        Exploration to find a new arch.
        """
        model = self._require_model()
        raw_arch = model.get_net(func=random.choice)
        new_arch = self._normalise_config(raw_arch)
        size, macs = self.calculate_flops_size(new_arch)
        size = int(size * 10) / 10.0
        macs = int(macs * 10) / 10.0
        return new_arch, size, macs

    def distillation(self):
        """
        Match the distribution of the outputs of subnets as the supernet.
        """
        model = self._require_model()
        model_factory = self._require_model_class()
        device = Config.device()
        model.train()
        criterion = torch.nn.KLDivLoss()
        # Cifar10 mean and std.
        mean = torch.tensor([0.0]).repeat(Config().trainer.batch_size, 3, 32, 32)
        std = torch.tensor([1.0]).repeat(Config().trainer.batch_size, 3, 32, 32)
        for rounds in range(Config.clients.per_round):
            model = model.to(device)
            self.model = model
            func = min if rounds == 0 else random.choice
            raw_config = model.get_net(func)
            subnet_config = self._normalise_config(raw_config)
            subnet = model_factory(
                configs=subnet_config, **Config().parameters.client_model._asdict()
            ).to(device)
            optimizer = torch.optim.Adam(
                subnet.parameters(),
                **Config().parameters.distillation.optimizer._asdict(),
            )
            for _ in range(
                int(
                    Config().parameters.distillation.iterations
                    / Config().trainer.batch_size
                )
            ):
                inputs = torch.normal(mean, std).to(device)
                subnet = subnet.to(device)
                with torch.no_grad():
                    soft_label = model(inputs)
                outputs = subnet(inputs)
                loss = criterion(outputs, soft_label)
                loss.backward()
                optimizer.step()
            subnet = subnet.cpu()
            model = model.cpu()
            self.model = model
            self.aggregation([subnet.state_dict()], update_track=False)
