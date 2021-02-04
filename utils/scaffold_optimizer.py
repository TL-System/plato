"""
A customized optimizer for SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
(https://arxiv.org/pdf/1910.06378.pdf)
"""
import torch
from torch import optim


class ScaffoldOptimizer(optim.SGD):
    """A customized optimizer for SCAFFOLD's local solver."""
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.c_server = None
        self.c_client = None
        self.c_plus = None
        self.flag_c_plus = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if self.flag_c_plus is True:
                self.c_plus = []

            # initialize c_server and c_client
            if self.c_server is None:
                self.c_client = [0] * len(group['params'])
                self.c_server = [0] * len(group['params'])

            for p, c_client, c_server in zip(group['params'], self.c_client,
                                             self.c_server):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply variance reduction
                d_p.add_(c_server)
                d_p.sub_(c_client)

                # update weight
                p.data.add_(-group['lr'], d_p)

                # update self.c_plus
                if self.flag_c_plus is True:
                    self.c_plus.append(d_p)
        return loss
