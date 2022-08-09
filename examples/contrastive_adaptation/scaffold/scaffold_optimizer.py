"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from collections import OrderedDict

import torch
from torch import optim


class ScaffoldOptimizer(optim.SGD):
    """A customized optimizer for SCAFFOLD."""
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.server_control_variate = OrderedDict()
        self.client_control_variate = OrderedDict()
        self.device = None

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = super().step(closure=closure)

        for group in self.param_groups:
            learning_rate = -group['lr']
            counter = 0
            for name in self.server_control_variate:
                if 'weight' in name or 'bias' in name:
                    server_control_variate = self.server_control_variate[
                        name].to(self.device)
                    param = group['params'][counter]
                    if self.client_control_variate:
                        param.data.add_(torch.sub(
                            server_control_variate,
                            self.client_control_variate[name].to(self.device)),
                                        alpha=learning_rate)
                    else:
                        param.data.add_(server_control_variate,
                                        alpha=learning_rate)
                    counter += 1

        return loss
