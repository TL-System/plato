"""
A customized optimizer for FedProx.

Reference:

Li et al., "Federated Optimization in Heterogeneous Networks."
(https://arxiv.org/pdf/1812.06127.pdf)

"""
import torch
from plato.config import Config
from torch import optim


class FedProxOptimizer(optim.SGD):
    """A customized optimizer for FedProx's local solver."""
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

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

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
                # apply proximal update
                d_p.add_(Config().trainer.mu, p.data - param_state['old_init'])
                p.data.add_(-group['lr'], d_p)

        return loss

    def params_state_update(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['old_init'] = torch.clone(p.data).detach()
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
