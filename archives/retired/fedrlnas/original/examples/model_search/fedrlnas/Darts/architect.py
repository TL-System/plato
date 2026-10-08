"""
Modify based on cnn/architect.py in https://github.com/quark0/darts,
to support the algorithms in FedRLNAS.
"""

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
        baseline_decay=0.99,
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
        self.optimizer = torch.optim.Adam(
            self.arch_parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
            weight_decay=weight_decay,
        )

        self.baseline = 1.0 / Config().parameters.model.num_classes
        if hasattr(Config().parameters.architect, "baseline_decay"):
            self.baseline_decay = Config().parameters.architect.baseline_decay
        else:
            self.baseline_decay = baseline_decay

    def forward(self, feature):
        "Forwards output of the supernet."
        return self.model(feature)

    def step(self, epoch_acc, epoch_index_normal, epoch_index_reduce):
        """Step of architect, update architecture parameters."""
        rewards = self._compute_reward(epoch_acc)
        normal_grad = self._compute_grad(
            self.alphas_normal, rewards, epoch_index_normal
        )
        reduce_grad = self._compute_grad(
            self.alphas_reduce, rewards, epoch_index_reduce
        )

        # Initialize gradients if they don't exist
        if self.alphas_normal.grad is None:
            self.alphas_normal.grad = torch.zeros_like(self.alphas_normal)
        if self.alphas_reduce.grad is None:
            self.alphas_reduce.grad = torch.zeros_like(self.alphas_reduce)

        self.alphas_normal.grad.copy_(normal_grad)
        self.alphas_reduce.grad.copy_(reduce_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _compute_reward(self, accuracy_list):
        avg_acc = torch.mean(torch.Tensor(accuracy_list))
        self.baseline += self.baseline_decay * (avg_acc - self.baseline)
        return [accuracy_list[i] - self.baseline for i in range(len(accuracy_list))]

    def _compute_grad(self, alphas, rewards, index_list):
        grad = torch.zeros(alphas.size())
        prob = F.softmax(alphas, dim=-1)
        for client_idx, reward in enumerate(rewards):
            reward = rewards[client_idx]
            index = index_list[client_idx]
            client_grad = torch.Tensor(prob.shape)
            client_grad.copy_(prob)
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
        """Returns architecture parameters."""
        return self._arch_parameters

    def genotype(self):
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

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self.model.multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
