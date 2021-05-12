"""
A customized trainer for FedSarah.

Reference: Ngunyen et al., "SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient" (https://arxiv.org/pdf/1703.00102.pdf) 
"""
import torch
from torch import optim
import os


class FedSarahOptimizer(optim.Adam):
    def __init__(self, params):
        super().__init__(params)
        self.server_control_variates = None
        self.client_control_variates = None
        self.client_id = None
        self.last_model_weights = None
        self.epoch_counter = 0
        self.max_counter = None

    def params_state_update(self):

        self.epoch_counter += 1
        new_client_control_variates = []
        new_last_model_weights = []

        if self.epoch_counter == 1:
            filename = f"last_model_weights_{self.client_id}.pth"
            if os.path.exists(filename):
                self.last_model_weights = torch.load(filename)

        for group in self.param_groups:

            #Initialize server control variates and client control variates
            if self.server_control_variates is None:
                self.client_control_variates = [0] * len(group['params'])
                self.server_control_variates = [0] * len(group['params'])
                self.last_model_weights = [0] * len(group['params'])

            for p, client_control_variate, server_control_variate, last_model_weight in zip(
                    group['params'], self.client_control_variates,
                    self.server_control_variates, self.last_model_weights):

                if p.grad is None:
                    continue

                # Compute control variates update
                control_variate_update = torch.sub(server_control_variate,
                                                   client_control_variate)
                # Reduce variance
                p.data.add_(0.0001, control_variate_update)

                # Obtain new control variates
                new_client_control_variates.append(p.data - last_model_weight)
                new_last_model_weights.append(p.data)

            # Update control variates
            self.client_control_variates = new_client_control_variates
            self.last_model_weights = new_last_model_weights

            # Save the updated client control variates and model weights for next rounds
            if self.epoch_counter == self.max_counter:
                fn = f"new_client_control_variates_{self.client_id}.pth"
                torch.save(self.client_control_variates, fn)

                fn = f"last_model_weights_{self.client_id}.pth"
                torch.save(self.last_model_weights, fn)


'''
may use later

class FedSarahOptimizer(optim.SGD):
    """A customized optimizer for FedSarah."""
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)

        self.server_control_variates = None
        self.client_control_variates = None
        self.client_id = None
        self.max_counter = None
        self.max_epsilon = 2
        self.min_epsilon = 1
        self.epsilon_decay = 0.1
        self.epsilon = None
        self.epoch_counter = 0

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

            # Initialize server control variates and client control variates
            if self.server_control_variates is None:
                self.client_control_variates = [0] * len(group['params'])
                self.server_control_variates = [0] * len(group['params'])

            for p, client_control_variate, server_control_variate in zip(
                    group['params'], self.client_control_variates,
                    self.server_control_variates):
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
                control_variate_batch = self.epsilon * torch.sub(
                    server_control_variate,
                    client_control_variate)  # will be adaptively adjusted
                control_variate_batch = torch.add(d_p, control_variate_batch)

                # Update weights
                p.data.add_(
                    -group['lr'],
                    control_variate_batch)  # will be adaptively adjusted

        return loss

    def params_state_update(self):
        """ Update control variates for every local step"""

        self.epoch_counter += 1
        new_client_control_variates = []
        new_server_control_variates = []

        for group in self.param_groups:
            for p, client_control_variate, server_control_variate in zip(
                    group['params'], self.client_control_variates,
                    self.server_control_variates):

                if p.grad is None:
                    continue

                # Compute server control variates
                server_control_variate_update = torch.sub(
                    server_control_variate, client_control_variate)
                server_control_variate_update = torch.add(
                    p.grad.data, server_control_variate_update)

                # Obtain new control variates
                new_client_control_variates.append(p.grad.data)
                new_server_control_variates.append(
                    server_control_variate_update)

            # Update control variates
            self.client_control_variates = new_client_control_variates
            self.server_control_variates = new_server_control_variates

            # Save the updated client control variates
            if self.epoch_counter == self.max_counter:
                fn = f"new_client_control_variates_{self.client_id}.pth"
                torch.save(self.client_control_variates, fn)
'''
