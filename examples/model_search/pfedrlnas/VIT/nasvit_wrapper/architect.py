"""
NAS architect in PerFedRLNAS, a wrapper over the supernet.
"""

import copy
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from plato.config import Config

from .attentive_nas_dynamic_model import AttentiveNasDynamicModel
from .config import _C as config


# pylint:disable=attribute-defined-outside-init
class Architect(nn.Module):
    """The supernet wrapper, including supernet and arch parameters."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
    ):
        super().__init__()
        self.initialization()

    def initialization(self):
        """
        Initialization function.
        """
        self.model = AttentiveNasDynamicModel(supernet=config.supernet_config)
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
            # Use model_path if available, otherwise use default models/pretrained directory
            if hasattr(Config().server, "model_path"):
                model_dir = Config().server.model_path
            else:
                model_dir = "./models/pretrained"
            save_config = f"{model_dir}/baselines.pickle"
            if os.path.exists(save_config):
                with open(save_config, "rb") as file:
                    self.baseline = pickle.load(file)
        self.lambda_time = Config().parameters.architect.lambda_time
        self.lambda_neg = Config().parameters.architect.lambda_neg
        self.device = Config().device()

    def step(self, epoch_reward, epoch_index, client_id_list):
        """Step of architect, update architecture parameters."""
        rewards = self._compute_reward(epoch_reward, client_id_list)
        if (
            hasattr(Config().parameters.architect, "natural")
            and Config().parameters.architect.natural
        ):
            grads = self._compute_grad_natural(rewards, epoch_index, client_id_list)
        else:
            grads = self._compute_grad(rewards, epoch_index, client_id_list)
        for index, client_id in enumerate(client_id_list):
            # Ensure gradient tensor exists before copying
            if self.alphas[client_id - 1].grad is None:
                self.alphas[client_id - 1].grad = torch.zeros_like(
                    self.alphas[client_id - 1]
                )
            self.alphas[client_id - 1].grad.copy_(grads[index])
            self.optimizers[client_id - 1].step()
            self.optimizers[client_id - 1].zero_grad()

    def _compute_grad(self, rewards, index_list, client_id_list):
        grads = []
        for client_idx, reward in enumerate(rewards):
            alpha = self.alphas[client_id_list[client_idx] - 1]
            grad = torch.zeros(alpha.size())
            for edge_idx in range(len(self.stop_index) - 1):
                prob = F.softmax(
                    alpha[
                        self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx + 1]
                        + 1
                    ],
                    dim=-1,
                )
                index = index_list[client_idx]
                client_grad = torch.Tensor(prob.shape)
                client_grad.copy_(prob)
                # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
                index_prob = client_grad[index[edge_idx]]
                client_grad[index[edge_idx]] = index_prob - 1
                grad[
                    self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx + 1] + 1
                ] += reward * client_grad
            grads.append(grad)
        return grads

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
        reward = (
            accuracy
            - avg_accuracy
            - self.lambda_time * (round_time - np.min(round_time))
            - self.lambda_neg * neg_ratio
        )
        _add_reward_into_baseline(self, accuracy_list, client_id_list)
        logging.info("reward: %s", str(reward))
        # self.baseline = (
        #     self.baseline_decay * np.mean(accuracy)
        #     + (1 - self.baseline_decay) * self.baseline
        # )
        return reward

    def _compute_grad_natural(self, rewards, index_list, client_id_list):
        # pylint:disable=too-many-locals
        grads = []
        for reward, client_idx in enumerate(rewards):
            alpha = self.alphas[client_id_list[client_idx] - 1]
            grad = torch.zeros(alpha.size())
            for edge_idx in range(len(self.stop_index) - 1):
                prob = F.softmax(
                    alpha[
                        self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx + 1]
                        + 1
                    ],
                    dim=-1,
                )
                index = index_list[client_idx]
                client_grad = torch.Tensor(prob.shape)
                client_grad.copy_(prob)
                # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
                for edge_idx in range(client_grad.shape[0]):
                    index_prob = client_grad[edge_idx][index[edge_idx]]
                    client_grad[edge_idx][index[edge_idx]] = index_prob - 1
                    dalpha = client_grad
                    finverse = torch.pinverse(
                        torch.matmul(dalpha, torch.transpose(dalpha, 0, 1))
                    )
                grad[self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx] + 1] += (
                    reward * torch.matmul(finverse, client_grad)
                )
            grads.append(grad)
        return grads

    def _initialize_alphas(self):
        # pylint:disable=too-many-locals
        cfg_candidate = self.model.cfg_candidates
        stop_index = [-1]
        resolution = cfg_candidate["resolution"]
        stop_index.append(len(resolution) - 1)
        width = cfg_candidate["width"]
        for width_candidate in width:
            stop_index.append(stop_index[-1] + len(width_candidate))
        depth = cfg_candidate["depth"]
        for depth_candidate in depth:
            stop_index.append(stop_index[-1] + len(depth_candidate))
        kernel_size = cfg_candidate["kernel_size"]
        for kernel_size_candidate in kernel_size:
            stop_index.append(stop_index[-1] + len(kernel_size_candidate))
        expand_ratio = cfg_candidate["expand_ratio"]
        for expand_ratio_candidate in expand_ratio:
            stop_index.append(stop_index[-1] + len(expand_ratio_candidate))

        self.stop_index = stop_index
        self.alphas = [
            Variable(
                torch.zeros(self.stop_index[-1] + 1),
                requires_grad=True,
            )
            for _ in range(self.client_nums)
        ]
        self._arch_parameters = [[alpha] for alpha in self.alphas]

        for alpha in self.alphas:
            feature = Variable(torch.Tensor(alpha.size()), requires_grad=False)
            logits = feature * alpha
            loss = torch.mean(logits)
            loss.backward()

        self.model.zero_grad()
        self.zero_grad()

    def arch_parameters(self):
        """Returns architecture parameters."""
        return self._arch_parameters

    def get_index(self, prob, length):
        """Exract the index of choice based on given probability."""
        prob = F.softmax(prob, dim=-1)
        prob = prob.detach().numpy()
        prob /= prob.sum()
        mask_idx = np.random.choice(range(length), 1, replace=False, p=prob)
        return mask_idx[0]

    def sample_config(self, client_id):
        """Smaple a ViT structure from current alphas"""
        cfg_candidate = self.model.cfg_candidates
        # pylint: disable=unsubscriptable-object
        alpha = self.alphas[client_id]
        cfg = {}
        stop_index_point = 1
        length = self.stop_index[stop_index_point] + 1
        prob = alpha[: self.stop_index[stop_index_point] + 1]
        cfg["resolution"] = cfg_candidate["resolution"][self.get_index(prob, length)]
        stop_index_point += 1
        for candidate_name in ["width", "depth", "kernel_size", "expand_ratio"]:
            candidate_list = []
            for candidate_candidate in cfg_candidate[candidate_name]:
                length = (
                    self.stop_index[stop_index_point]
                    - self.stop_index[stop_index_point - 1]
                )
                prob = alpha[
                    self.stop_index[stop_index_point - 1] + 1 : self.stop_index[
                        stop_index_point
                    ]
                    + 1
                ]
                candidate_list.append(candidate_candidate[self.get_index(prob, length)])
                stop_index_point += 1
            cfg[candidate_name] = candidate_list
        return cfg

    def extract_index(self, subnets_config):
        """Exract the index with which choice is made in each candidate pool."""
        cfg_candidate = self.model.cfg_candidates
        epoch_index = []
        for cfg in subnets_config:
            index = []
            resolution = cfg_candidate["resolution"]
            index.append(resolution.index(cfg["resolution"]))
            for candidate_name in ["width", "depth", "kernel_size", "expand_ratio"]:
                candidate = cfg_candidate[candidate_name]
                for candidate, candidate_list in zip(cfg[candidate_name], candidate):
                    index.append(candidate_list.index(candidate))
            epoch_index.append(index)
        return epoch_index

    def forward(self, feature):
        "Forwards output of the supernet."
        return self.model(feature)
