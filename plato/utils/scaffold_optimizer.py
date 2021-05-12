"""
A customized optimizer for SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
(https://arxiv.org/pdf/1910.06378.pdf)
"""
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
        self.new_client_update_direction = None
        self.server_update_direction = None
        self.client_update_direction = None
        self.client_id = None

        self.update_flag = True

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

            if self.update_flag is True:
                self.new_client_update_direction = []

            # Initialize server update direction and client update direction
            if self.server_update_direction is None:
                self.client_update_direction = [0] * len(group['params'])
                self.server_update_direction = [0] * len(group['params'])

            for p, client_update_direction, server_update_direction in zip(
                    group['params'], self.client_update_direction,
                    self.server_update_direction):
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

                # Apply variance reduction
                d_p.add_(server_update_direction)
                d_p.sub_(client_update_direction)

                # Update weight
                p.data.add_(-group['lr'], d_p)

                # Obtain the latest client update direction
                if self.update_flag is True:
                    self.new_client_update_direction.append(d_p)

        if self.update_flag is True:
            fn = f"new_client_update_direction_{self.client_id}.pth"
            torch.save(self.new_client_update_direction, fn)
            self.update_flag = False

        return loss
