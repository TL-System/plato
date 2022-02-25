"""
A federated learning training session using FedProx.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems, 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""
import torch

from plato.config import Config
from plato.trainers import basic


class FedProxLocalObjective():
    """ Representing the local objective of FedProx clients. """

    def __init__(self, model):
        self.model = model
        self.init_global_weights = torch.tensor([], requires_grad=False)
        for param in model.parameters():
            self.init_global_weights = torch.cat(
                (self.init_global_weights, torch.flatten(param)))

    def compute_objective(self, outputs, labels):
        """ Compute the objective the FedProx client wishes to minimize. """
        cur_weights = torch.tensor([], requires_grad=False)
        for param in self.model.parameters():
            cur_weights = torch.cat((cur_weights, torch.flatten(param)))

        mu = Config().clients.proximal_term_penalty_constant
        prox_term = mu / 2 * torch.linalg.norm(
            cur_weights - self.init_global_weights, ord=2)

        local_function = torch.nn.CrossEntropyLoss()
        h = local_function(outputs, labels) + prox_term
        return h


class Trainer(basic.Trainer):
    """ The federated learning trainer for the FedProx client. """

    def loss_criterion(self, model):
        """ Return the loss criterion for FedProx clients. """
        local_obj = FedProxLocalObjective(model)
        return local_obj.compute_objective
