"""
A federated learning training session using FedProx.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems, 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""
import torch
import numpy as np

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

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in FedProx framework. """
        np.random.seed(self.client_id)
        # Determine whether this selected client is a straggler
        strag_prop = Config().clients.straggler_percentage / 100
        is_straggler = np.random.choice([True, False],
                                        p=[strag_prop, 1 - strag_prop])
        if is_straggler:
            # Choose the epoch uniformly as mentioned in Section 5.2 of the paper
            global_epochs = Config().trainer.epochs
            config['epochs'] = np.random.choice(np.arange(1, global_epochs))

        super(Trainer, self).train_process(config, trainset, sampler,
                                           cut_layer)

    def loss_criterion(self, model):
        """ Return the loss criterion for FedProx clients. """
        local_obj = FedProxLocalObjective(model)
        return local_obj.compute_objective
