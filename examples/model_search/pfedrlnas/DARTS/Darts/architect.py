"""
Modify based on cnn/architect.py in https://github.com/quark0/darts,
to support the algorithms in FedRLNAS.
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from Darts.genotypes import PRIMITIVES, Genotype
from Darts.model_search import Network
from plato.config import Config


class Architect(nn.Module):
    """The supernet wrapper, including supernet and arch parameters."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        learning_rate=3e-3,
        weight_decay=3e-1,
    ):
        super().__init__()
        self.model = Network()
        if hasattr(Config().parameters.architect, "pretrain_path"):
            self.model.load_state_dict(
                torch.load(Config().parameters.architect.pretrain_path)
            )
        self.client_nums = Config().clients.total_clients
        self._initialize_alphas()

        if hasattr(Config().parameters.architect, "learning_rate"):
            learning_rate = Config().parameters.architect.learning_rate
        if hasattr(Config().parameters.architect, "weight_decay"):
            weight_decay = Config().parameters.architect.weight_decay
        self.optimizers = [
            torch.optim.Adam(
                alpha,
                lr=learning_rate,
                betas=(0.5, 0.999),
                weight_decay=weight_decay,
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
        self.device = Config().device()

    def forward(self, feature):
        "Forwards output of the supernet."
        return self.model(feature)

    def step(
        self, rewards_list, epoch_index_normal, epoch_index_reduce, client_id_list
    ):
        """Step of architect, update architecture parameters."""
        rewards = self._compute_reward(rewards_list, client_id_list)
        normal_grads = self._compute_grad(rewards, epoch_index_normal, client_id_list)
        reduce_grads = self._compute_grad(rewards, epoch_index_reduce, client_id_list)
        for index, client_id in enumerate(client_id_list):
            # Ensure gradient tensors exist before copying
            if self.alphas_normal[client_id - 1].grad is None:
                self.alphas_normal[client_id - 1].grad = torch.zeros_like(
                    self.alphas_normal[client_id - 1]
                )
            if self.alphas_reduce[client_id - 1].grad is None:
                self.alphas_reduce[client_id - 1].grad = torch.zeros_like(
                    self.alphas_reduce[client_id - 1]
                )
            self.alphas_normal[client_id - 1].grad.copy_(normal_grads[index])
            self.alphas_reduce[client_id - 1].grad.copy_(reduce_grads[index])
            self.optimizers[client_id - 1].step()
            self.optimizers[client_id - 1].zero_grad()

    def _compute_reward(self, reward_list, client_id_list):
        # scale accuracy to 0-1
        accuracy_list = reward_list[0]
        round_time_list = reward_list[1]
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
        )
        _add_reward_into_baseline(self, accuracy_list, client_id_list)
        logging.info("reward: %s", str(reward))
        # self.baseline = (
        #     self.baseline_decay * np.mean(accuracy)
        #     + (1 - self.baseline_decay) * self.baseline
        # )
        return reward

    def _compute_grad(self, rewards, index_list, client_id_list, normal=True):
        # pylint: disable=too-many-locals
        grads = []
        for list_index, client_id in enumerate(client_id_list):
            if normal:
                alphas = self.alphas_normal[client_id - 1]
            else:
                alphas = self.alphas_reduce[client_id - 1]
            grad = torch.zeros(alphas.size())
            prob = F.softmax(alphas, dim=-1)
            reward = rewards[list_index]
            index = index_list[list_index]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
            for edge_idx in range(client_grad.shape[0]):
                index_prob = client_grad[edge_idx][index[edge_idx]]
                client_grad[edge_idx][index[edge_idx]] = index_prob - 1
            grad += reward * client_grad
            grads.append(grad)
        return grads

    def _initialize_alphas(self):
        self._steps = 4
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = [
            Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
            for _ in range(self.client_nums)
        ]
        self.alphas_reduce = [
            Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
            for _ in range(self.client_nums)
        ]
        self._arch_parameters = [
            [self.alphas_normal[i], self.alphas_reduce[i]]
            for i in range(self.client_nums)
        ]

        for alpha in self._arch_parameters:
            normal = alpha[0]
            reduce = alpha[1]
            alpha = normal
            feature = Variable(torch.Tensor(alpha.size()), requires_grad=False)
            logits = feature * alpha
            loss = torch.mean(logits)
            loss.backward()
            alpha = reduce
            feature = Variable(torch.Tensor(alpha.size()), requires_grad=False)
            logits = feature * alpha
            loss = torch.mean(logits)
            loss.backward()

        self.model.zero_grad()
        self.zero_grad()

    def arch_parameters(self):
        """Returns architecture parameters."""
        return self._arch_parameters

    def genotype(self, alphas_normal, alphas_reduce):
        """Generates Genotypes."""

        def _parse(weights):
            gene = []
            n_index = 2
            start = 0
            for i in range(self._steps):
                end = start + n_index
                weight_matrix = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x, wm=weight_matrix: -max(
                        wm[x][k]
                        for k in range(len(wm[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(weight_matrix[j])):
                        if k != PRIMITIVES.index("none"):
                            if (
                                k_best is None
                                or weight_matrix[j][k] > weight_matrix[j][k_best]
                            ):
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n_index += 1
            return gene

        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self.model.multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
