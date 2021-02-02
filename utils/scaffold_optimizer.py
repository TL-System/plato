"""
A customized optimizer for SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (https://arxiv.org/pdf/1910.06378.pdf)
"""
import torch
from torch import optim
from config import Config


class ScaffoldOptimizer(optim.SGD):
    """A customized optimizer for SCAFFOLD's local solver."""
    def __init__(self, c_client, c_server):
        super.__init__():
        self.c_server = None
        self.c_client = None

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

            self.c_plus = []
            for p in group['params']:
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
                if self.c_client != 0:
                    d_p.add_(-self.c_client, self.c_server)

                p.data.add_(-group['lr'], d_p)

                # update self.c_plus 
                # self.c_plus.append(d_p)
