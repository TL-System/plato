import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from plato.config import Config
from .model_search import Network
from .genotypes import PRIMITIVES
from plato.config import Config


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
        self._initialize_alphas()
        self.optimizer = torch.optim.Adam(
            self.arch_parameters(),
            lr=Config().parameters.architect.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=Config().parameters.architect.weight_decay,
        )
        self.baseline = 1.0 / Config().parameters.model.num_classes
        self.baseline_decay = Config().parameters.architect.baseline_decay

    def step(self, epoch_acc, epoch_index_normal, epoch_index_reduce):
        rewards = self._compute_reward(epoch_acc)
        normal_grad = self._compute_grad(
            self.alphas_normal, rewards, epoch_index_normal
        )
        reduce_grad = self._compute_grad(
            self.alphas_reduce, rewards, epoch_index_reduce
        )
        self.alphas_normal.grad.copy_(normal_grad)
        self.alphas_reduce.grad.copy_(reduce_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _compute_reward(self, accuracy_list):
        # scale accuracy to 0-1
        avg_acc = torch.mean(torch.Tensor(accuracy_list))
        self.baseline += self.baseline_decay * (avg_acc - self.baseline)
        # reward = accuracy - baseline
        return [accuracy_list[i] - self.baseline for i in range(len(accuracy_list))]

    def _compute_grad(self, alphas, rewards, index_list):
        grad = torch.zeros(alphas.size())
        prob = F.softmax(alphas, dim=-1)
        for client_idx, reward in enumerate(rewards):
            reward = rewards[client_idx]
            index = index_list[client_idx]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
            # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
            for edge_idx in range(client_grad.shape[0]):
                index_prob = client_grad[edge_idx][index[edge_idx]]
                client_grad[edge_idx][index[edge_idx]] = index_prob - 1
            grad += reward * client_grad
        grad /= len(rewards)
        return grad

    def _initialize_alphas(self):
        self._steps = 4
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops), requires_grad=True
        )
        self.alphas_reduce = Variable(
            1e-3 * torch.randn(k, num_ops), requires_grad=True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        inputs = Variable(torch.Tensor(self.alphas_normal.size()), requires_grad=False)
        logits = inputs * self.alphas_normal
        loss = torch.mean(logits)
        loss.backward()

        inputs = Variable(torch.Tensor(self.alphas_reduce.size()), requires_grad=False)
        logits = inputs * self.alphas_reduce
        loss = torch.mean(logits)
        loss.backward()

        self.model.zero_grad()
        self.zero_grad()

    def arch_parameters(self):
        return self._arch_parameters
