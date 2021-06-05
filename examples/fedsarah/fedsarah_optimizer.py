"""
A customized optimizer for FedSarah.

Reference: Ngunyen et al., "SARAH: A Novel Method for Machine Learning Problems
Using Stochastic Recursive Gradient." (https://arxiv.org/pdf/1703.00102.pdf)
"""
import os

import torch
from torch import optim


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

            # Initialize server control variates and client control variates
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
                p.data.add_(control_variate_update, alpha=0.0001)

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
