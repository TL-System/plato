"""
FedTP learns a personalized self-attention layer for each client
while the parameters of the other layers are shared among the clients.

Reference:
Li, Hongxia, Zhongyi Cai, Jingya Wang, Jiangnan Tang, Weiping Ding, Chin-Teng Lin, and Ye Shi.
"FedTP: Federated Learning by Transformer Personalization."
arXiv preprint arXiv:2211.01572 (2022).

https://arxiv.org/pdf/2211.01572v1.pdf.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from plato.algorithms import fedavg
from plato.config import Config
from plato.trainers.basic import Trainer


class ServerAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for the FedTP on server."""

    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.current_weights: Tensor | None = None

    def generate_attention(self, hnet: Module, client_id: int) -> Tensor:
        """Generated the customized attention of each client."""
        weights = hnet(
            torch.tensor([client_id - 1], dtype=torch.long).to(Config().device())
        )
        self.current_weights = weights
        return weights

    def calculate_hnet_grads(
        self,
        node_weights: dict[str, Tensor],
        delta_theta: dict[str, Tensor],
        hnet: Module,
    ) -> Tuple[Tensor, ...]:
        """Manullay calculate the gradients of hypernet."""
        hnet_grads = torch.autograd.grad(
            list(node_weights.values()),
            hnet.parameters(),
            grad_outputs=list(delta_theta.values()),
            retain_graph=True,
        )
        return hnet_grads

    def get_hnet_optimizer(self, hnet: Module) -> Optimizer:
        """Get the specific optimizer of hypernet."""
        optimizer = torch.optim.SGD(
            hnet.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3
        )
        return optimizer
