"""
A federated learning training session using FedProx.

To better handle system heterogeneity, the FedProx algorithm introduced a
proximal term in the optimizer used by local training on the clients. It has
been quite widely cited and compared with in the federated learning literature.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems (MLSys), vol. 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""
import torch

from plato.config import Config
from plato.trainers import basic


def _flatten_weights_from_model(model, device):
    """Return the weights of the given model as a 1-D tensor"""
    weights = torch.tensor([], requires_grad=False).to(device)
    model.to(device)
    for param in model.parameters():
        weights = torch.cat((weights, torch.flatten(param)))
    return weights


class FedProxLocalObjective:
    """Representing the local objective of FedProx clients."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.init_global_weights = _flatten_weights_from_model(model, device)

    def compute_objective(self, outputs, labels):
        """Compute the objective the FedProx client wishes to minimize."""
        current_weights = _flatten_weights_from_model(self.model, self.device)
        parameter_mu = (
            Config().clients.proximal_term_penalty_constant
            if hasattr(Config().clients, "proximal_term_penalty_constant")
            else 1
        )
        proximal_term = (
            parameter_mu
            / 2
            * torch.linalg.norm(current_weights - self.init_global_weights, ord=2)
        )

        local_function = torch.nn.CrossEntropyLoss()
        function_h = local_function(outputs, labels) + proximal_term
        return function_h


class Trainer(basic.Trainer):
    """The federated learning trainer for the FedProx client."""

    def get_loss_criterion(self):
        """Return the loss criterion for FedProx clients."""
        local_obj = FedProxLocalObjective(self.model, self.device)
        return local_obj.compute_objective
