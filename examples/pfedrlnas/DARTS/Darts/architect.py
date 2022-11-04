import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from .stale import compute_stale_grad_alpha
from plato.config import Config
from .model_search import Network
from .genotypes import PRIMITIVES
from .valuenet import ValueNet
from plato.config import Config
import logging


class Architect(nn.Module):
    def __init__(
        self,
        model=None,
        momentum=0.9,
        weight_decay=3e-4,
        arch_learning_rate=3e-3,
        arch_weight_decay=3e-1,
        arch_baseline_decay=0.99,
    ):
        super().__init__()
        self.model = Network()
        if hasattr(Config().parameters.architect, "pretrain_path"):
            self.model.load_state_dict(
                torch.load(Config().parameters.architect.pretrain_path)
            )
        self.client_nums = Config().clients.total_clients
        if (
            hasattr(Config().parameters.architect, " personalize_last")
            and Config().parameters.architect.personalize_last
        ):
            self.lasts = [
                copy.deepcopy(self.model.classifier) for _ in range(self.client_nums)
            ]
        self._initialize_alphas()
        self.optimizer = torch.optim.Adam(
            self.arch_parameters(),
            lr=Config().parameters.architect.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=Config().parameters.architect.weight_decay,
        )
        # self.optimizer=torch.optim.SGD(self.arch_parameters(),lr=0.5,momentum=0.9)
        self.baseline = 1.0 / Config().parameters.model.num_classes
        self.baseline_decay = Config().parameters.architect.baseline_decay
        self.device = Config().device()
        # value net
        if hasattr(Config().parameters.architect, "value_net"):
            k = sum(1 for i in range(self.model._steps) for _ in range(2 + i))
            num_ops = len(PRIMITIVES)
            self.value_net = ValueNet(k * num_ops * 2)
            self.value_net = self.value_net.train()
            self.value_net = self.value_net.to(self.device)

    def step(self, epoch_acc, epoch_index_normal, epoch_index_reduce, client_id_list):
        rewards = self._compute_reward(epoch_acc, client_id_list)
        if (
            hasattr(Config().parameters.architect, "natural")
            and Config().parameters.architect.natural
        ):
            grad_normal = self._compute_grad_natural(
                self.alphas_normal, rewards, epoch_index_normal, client_id_list
            )
            grad_reduce = self._compute_grad_natural(
                self.alphas_reduce, rewards, epoch_index_reduce, client_id_list
            )
        else:
            grad_normal = self._compute_grad(
                self.alphas_normal, rewards, epoch_index_normal, client_id_list
            )
            grad_reduce = self._compute_grad(
                self.alphas_reduce, rewards, epoch_index_reduce, client_id_list
            )
        logging.info(str(grad_normal[client_id_list[0] - 1][0]))
        self.alphas_normal.grad.copy_(grad_normal)
        self.alphas_reduce.grad.copy_(grad_reduce)
        self.optimizer.step()
        self.optimizer.zero_grad()
        logging.info(str(self.alphas_normal[client_id_list[0] - 1][0]))
        logging.info(str(F.softmax(self.alphas_normal[client_id_list[0] - 1][0])))

    def _compute_grad(self, alphas, rewards, index_list, client_id_list):
        grad = torch.zeros(alphas.size())
        for client_idx in range(len(rewards)):
            prob = F.softmax(alphas[client_id_list[client_idx] - 1], dim=-1)
            reward = rewards[client_idx]
            index = index_list[client_idx]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
            # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
            for edge_idx in range(client_grad.shape[0]):
                index_prob = client_grad[edge_idx][index[edge_idx]]
                client_grad[edge_idx][index[edge_idx]] = index_prob - 1
            grad[client_id_list[client_idx] - 1] += reward * client_grad
            # grad /= len(rewards)
        return grad

    def _compute_reward(self, accuracy_list, client_id_list):
        # scale accuracy to 0-1
        accuracy = np.array(accuracy_list)
        if hasattr(Config().parameters.architect, "value_net"):
            reward = torch.from_numpy(accuracy)
            alphas = torch.stack(
                [
                    self.alphas_normal[torch.tensor(client_id_list) - 1],
                    self.alphas_reduce[torch.tensor(client_id_list) - 1],
                ],
                dim=-1,
            )
            alphas, reward = alphas.to(self.device), reward.to(self.device)
            v = self.value_net.forward(alphas)
            v = v[:, 0]
            least_value = 1.0 / self.client_nums
            v = torch.where(
                v < least_value, torch.ones(v.size()).to(self.device) * least_value, v
            )
            reward = reward - v
            return reward.detach().cpu()
        else:
            reward = accuracy - self.baseline
            logging.info("reward: %s", str(reward))
            self.baseline = (
                self.baseline_decay * accuracy
                + (1 - self.baseline_decay) * self.baseline
            )
            return reward

    def _compute_grad_natural(self, alphas, accuracy_list, index_list, client_id_list):
        grad = torch.zeros(alphas.size())
        rewards = self._compute_reward(accuracy_list, client_id_list)
        for client_idx in range(len(rewards)):
            prob = F.softmax(alphas[client_id_list[client_idx] - 1], dim=-1)
            reward = rewards[client_idx]
            index = index_list[client_idx]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
            # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
            for edge_idx in range(client_grad.shape[0]):
                index_prob = client_grad[edge_idx][index[edge_idx]]
                client_grad[edge_idx][index[edge_idx]] = index_prob - 1
            dalpha = client_grad
            Fish = torch.matmul(dalpha, torch.transpose(dalpha, 0, 1))
            Finverse = torch.pinverse(Fish)
            grad[client_id_list[client_idx] - 1] += reward * torch.matmul(
                Finverse, client_grad
            )
            # grad /= len(rewards)
        return grad

    def stale_step(
        self,
        epoch_acc,
        epoch_index_normal,
        epoch_index_reduce,
        stale_alphas_normal,
        stale_alphas_reduce,
        stale_acc,
        stale_index_normal,
        stale_index_reduce,
    ):
        self._compute_stale_grad(
            self.alphas_normal,
            epoch_acc,
            epoch_index_normal,
            stale_alphas_normal,
            stale_acc,
            stale_index_normal,
        )
        self._compute_stale_grad(
            self.alphas_reduce,
            epoch_acc,
            epoch_index_reduce,
            stale_alphas_reduce,
            stale_acc,
            stale_index_reduce,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _compute_stale_grad(
        self, alphas, accuracy_list, index_list, old_alphas, old_accuracy, old_index
    ):
        grad = torch.zeros(alphas.size())
        prob = F.softmax(alphas, dim=-1)
        rewards = self._compute_reward(accuracy_list)
        for client_idx in range(len(rewards)):
            reward = rewards[client_idx]
            index = index_list[client_idx]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
            # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
            for edge_idx in range(client_grad.shape[0]):
                index_prob = client_grad[edge_idx][index[edge_idx]]
                client_grad[edge_idx][index[edge_idx]] = index_prob - 1
            grad += reward * client_grad

        # stale update
        old_reward = self._compute_reward(old_accuracy)
        for stale_idx in range(len(old_alphas)):
            stale_grad = compute_stale_grad_alpha(
                old_index[stale_idx], old_alphas[stale_idx], alphas
            )
            grad += old_reward[stale_idx] * stale_grad
        grad /= len(rewards) + len(old_alphas)
        alphas.grad = grad

    def _initialize_alphas(self):
        self._steps = 4
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.zeros(self.client_nums, k, num_ops), requires_grad=True
        )
        self.alphas_reduce = Variable(
            1e-3 * torch.zeros(self.client_nums, k, num_ops), requires_grad=True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        input = Variable(torch.Tensor(self.alphas_normal.size()), requires_grad=False)
        logits = input * self.alphas_normal
        loss = torch.mean(logits)
        loss.backward()

        input = Variable(torch.Tensor(self.alphas_reduce.size()), requires_grad=False)
        logits = input * self.alphas_reduce
        loss = torch.mean(logits)
        loss.backward()

        self.model.zero_grad()
        self.zero_grad()

    def arch_parameters(self):
        return self._arch_parameters


if __name__ == "__main__":
    dalpha = torch.randn(8)
    # dalpha=F.softmax(dalpha)
    # dalpha[3]=1-dalpha[3]
    # dalpha=-dalpha
    dalpha = torch.reshape(dalpha, (dalpha.shape[0], 1))
    g = dalpha * 0.5
    Fish = torch.matmul(dalpha, torch.transpose(dalpha, 0, 1))
    Finverse = torch.inverse(Fish)
    x = torch.matmul(torch.transpose(g, 0, 1), g)
    print(x)
    Trust_grad = torch.sqrt(
        2 * 0.01 / torch.matmul(torch.matmul(torch.transpose(g, 0, 1), Finverse), g)
    ) * torch.matmul(Finverse, g)
    print(Trust_grad)
