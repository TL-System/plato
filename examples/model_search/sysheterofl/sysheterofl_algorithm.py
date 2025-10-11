"""
The algorithm for paper system-heterogenous federated learning through architecture search.
"""

import copy
import pickle
import random
import sys

import ptflops
import torch

from plato.algorithms import fedavg
from plato.config import Config


# pylint:disable=too-many-instance-attributes
class Algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the ElasticArch algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_config = None
        self.model_class = None
        self.epsilon = Config().parameters.limitation.epsilon  # 0.8
        self.max_loop = Config().parameters.limitation.max_loop  # 50
        self.configs = []
        self.arch_list = {}
        self.min_configs_size_flops = None
        self.size_flops_counts_dict = []
        self.arch_counts = 0
        self.min_configs = []
        self.biggest_net = None

    def extract_weights(self, model=None):
        self.model = self.model.cpu()
        payload = self.get_local_parameters()
        return payload

    def initialize_arch_map(self, model_class):
        """
        First add largest net into the map.
        """
        self.model_class = model_class
        if Config().parameters.supernet.width and Config().parameters.supernet.depth:
            self.model.channel_rate_lists = [[1], [1], [1], [1]]
        for func in [min, max]:
            config = self.model.get_net(func)
            size, macs = self.calculate_flops_size(config)
            self.add_into_config(config, size, macs)
        if Config().parameters.supernet.width and Config().parameters.supernet.depth:
            self.model.channel_rate_lists = [[0.5, 1], [0.5, 1], [1], [1]]

    def add_into_config(self, config, size, macs):
        """
        Add the configuration into current arch database.
        """
        self.arch_list[self.arch_counts] = config
        self.size_flops_counts_dict.append([size, macs, 0])
        self.sort_config(size, macs)
        self.arch_counts += 1

    def sort_config(self, size, flops):
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

    def calculate_flops_size(self, config):
        """
        Calculate the size and flops.
        """
        pre_model = self.model_class(
            configs=config, **Config().parameters.client_model._asdict()
        )
        size = sys.getsizeof(pickle.dumps(pre_model.state_dict())) / 1024**2
        macs, _ = ptflops.get_model_complexity_info(
            pre_model,
            (3, 32, 32),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        macs /= 1024**2
        size = int(size * 10) / 10.0
        macs = int(macs * 10) / 10.0
        return size, macs

    def get_local_parameters(self):
        """
        Get the parameters of local models from the global model.
        """
        current_config = self.current_config
        pre_model = self.model_class(
            configs=current_config, **Config().parameters.client_model._asdict()
        )
        local_parameters = pre_model.state_dict()

        for key, value in self.model.state_dict().items():
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
                        value.data[: local_parameters[key].shape[0],]
                    )
        pre_model = self.model_class(
            configs=current_config, **Config().parameters.client_model._asdict()
        )
        pre_model.load_state_dict(local_parameters)
        local_parameters = pre_model.state_dict()
        return local_parameters

    def aggregation(self, weights_received, update_track=True):
        """
        Aggregate weights of different complexities.
        """
        global_parameters = copy.deepcopy(self.model.state_dict())
        for key, value in self.model.state_dict().items():
            value_org = copy.deepcopy(value)
            count = torch.zeros(value.shape)
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
                        if update_track or (
                            not update_track
                            and not ("running" in key or "tracked" in key)
                        ):
                            global_parameters[key][: local_weights[key].shape[0]] += (
                                copy.deepcopy(local_weights[key])
                            )
                            count[: local_weights[key].shape[0]] += torch.ones(
                                local_weights[key].shape
                            )
            count = torch.where(count == 0, torch.ones(count.shape), count)
            global_parameters[key] = torch.div(
                global_parameters[key] - value_org, count
            )
        return global_parameters

    # pylint:disable=too-many-locals
    def choose_config(self, limitation):
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
        self.current_config = self.arch_list[current_config]
        self.size_flops_counts_dict[current_config][2] += 1
        self.update_biggest(self.current_config)
        return self.current_config

    def update_biggest(self, new_config):
        "Update the configuration of the biggest net."
        if self.biggest_net is None:
            self.biggest_net = new_config
        else:
            for config_index, biggest_net_config in enumerate(self.biggest_net):
                for index, (old, new) in enumerate(
                    zip(biggest_net_config, new_config[config_index])
                ):
                    self.biggest_net[config_index][index] = max(old, new)

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
        new_arch = self.model.get_net(func=random.choice)
        size, macs = self.calculate_flops_size(new_arch)
        size = int(size * 10) / 10.0
        macs = int(macs * 10) / 10.0
        return new_arch, size, macs

    def distillation(self):
        """
        Match the distribution of the outputs of subnets as the supernet.
        """
        self.model.train()
        criterion = torch.nn.KLDivLoss()
        # Cifar10 mean and std.
        mean = torch.tensor([0.0]).repeat(Config().trainer.batch_size, 3, 32, 32)
        std = torch.tensor([1.0]).repeat(Config().trainer.batch_size, 3, 32, 32)
        for rounds in range(Config.clients.per_round):
            self.model = self.model.to(Config.device())
            func = min if rounds == 0 else random.choice
            subnet_config = self.model.get_net(func)
            subnet = self.model_class(
                configs=subnet_config, **Config().parameters.client_model._asdict()
            ).to(Config.device())
            optimizer = torch.optim.Adam(
                subnet.parameters(),
                **Config().parameters.distillation.optimizer._asdict(),
            )
            for _ in range(
                int(
                    Config.parameters.distillation.iterations
                    / Config().trainer.batch_size
                )
            ):
                inputs = torch.normal(mean, std).to(Config.device())
                subnet = subnet.to(Config.device())
                with torch.no_grad():
                    soft_label = self.model(inputs)
                outputs = subnet(inputs)
                loss = criterion(outputs, soft_label)
                loss.backward()
                optimizer.step()
            subnet = subnet.cpu()
            self.model = self.model.cpu()
            self.aggregation([subnet.state_dict()], update_track=False)
