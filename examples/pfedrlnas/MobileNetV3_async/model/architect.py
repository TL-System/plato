"""
NAS architect in PerFedRLNAS, a wrapper over the supernet.
"""
import copy
import pickle
import sys
import os
import logging

import numpy as np
import torch
from plato.config import Config

from .mobilenetv3_supernet import NasDynamicModel
from .config import get_config

sys.path.append("./examples/pfedrlnas/")
from VIT.nasvit_wrapper import architect


# pylint:disable=too-few-public-methods
# pylint:disable=attribute-defined-outside-init
class Architect(architect.Architect):
    """The supernet wrapper, including supernet and arch parameters."""

    def __init__(
        self,
    ):
        super().__init__()
        self.initialization()
        self.min_round_time = 0

    # pylint: disable=too-many-instance-attributes
    def initialization(self):
        """
        Initizalization function.
        """
        self.model = NasDynamicModel(supernet=get_config().supernet_config)
        if hasattr(Config().parameters.architect, "pretrain_path"):
            weight = torch.load(
                Config().parameters.architect.pretrain_path, map_location="cpu"
            )["model"]
            del weight["classifier.linear.linear.weight"]
            del weight["classifier.linear.linear.bias"]
            self.model.load_state_dict(weight, strict=False)

        self.client_nums = Config().clients.total_clients
        if (
            hasattr(Config().parameters.architect, " personalize_last")
            and Config().parameters.architect.personalize_last
        ):
            self.lasts = [
                copy.deepcopy(self.model.classifier) for _ in range(self.client_nums)
            ]
        self._initialize_alphas()
        self.optimizers = [
            torch.optim.Adam(
                alpha,
                lr=Config().parameters.architect.learning_rate,
                betas=(0.5, 0.999),
                weight_decay=Config().parameters.architect.weight_decay,
            )
            for alpha in self.arch_parameters()
        ]
        self.baseline = {}
        if Config().args.resume:
            save_config = f"{Config().server.model_path}/baselines.pickle"
            if os.path.exists(save_config):
                with open(save_config, "rb") as file:
                    self.baseline = pickle.load(file)
        self.lambda_time = Config().parameters.architect.lambda_time
        self.lambda_neg = Config().parameters.architect.lambda_neg
        self.device = Config().device()

    def _compute_reward(self, reward_list, client_id_list):
        # scale accuracy to 0-1
        accuracy_list = reward_list[0]
        round_time_list = reward_list[1]
        neg_ratio = reward_list[2]
        accuracy = np.array(accuracy_list)
        round_time = np.array(round_time_list)

        def _add_reward_into_baseline(self, accuracy_list, client_id_list):
            for client_id, accuracy in zip(client_id_list, accuracy_list):
                if client_id in self.baseline:
                    self.baseline[client_id] = max(self.baseline[client_id], accuracy)
                else:
                    self.baseline[client_id] = accuracy

        if not self.baseline:
            _add_reward_into_baseline(self, accuracy_list, client_id_list)
            # self.baseline = np.mean(accuracy)
        avg_accuracy = np.mean(np.array([item[1] for item in self.baseline.items()]))
        self._update_min_round_time(round_time)
        reward = (
            accuracy
            - avg_accuracy
            - self.lambda_time * (round_time - self.min_round_time)
            - self.lambda_neg * neg_ratio
        )
        _add_reward_into_baseline(self, accuracy_list, client_id_list)
        logging.info("reward: %s", str(reward))
        return reward

    def _update_min_round_time(self, round_time):
        if self.min_round_time == 0:
            self.min_round_time = np.min(round_time)
        else:
            self.min_round_time = min(self.min_round_time, np.min(round_time))
