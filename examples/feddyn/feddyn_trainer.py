"""
An implementation of the FedDyn algorithm.

D. Acar, et al., "Federated Learning Based on Dynamic Regularization," in the
Proceedings of ICLR 2021.

Paper: https://openreview.net/forum?id=B7v4QMR6Z9w

Source code: https://github.com/alpemreacar/FedDyn
"""
import copy
import os
import torch
import numpy as np

from plato.config import Config
from plato.trainers import basic


# pylint:disable=no-member
# pylint:disable=too-many-instance-attributes
class Trainer(basic.Trainer):
    """
    FedDyn's Trainer.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.server_model_param = None
        self.local_param_list = None

    # pylint:disable=too-many-locals
    def perform_forward_and_backward_passes(
        self,
        config,
        examples,
        labels,
    ):
        """Perform forward and backward passes in the training loop."""
        labels = labels.cpu().numpy()
        weight_list = labels / np.sum(labels) * Config().clients.total_clients

        alpha_coef = (
            Config().algorithm.alpha_coef
            if hasattr(Config().algorithm, "alpha_coef")
            else 0.01
        )
        adaptive_alpha_coef = alpha_coef / np.where(weight_list != 0, weight_list, 1.0)

        server_model_param = self.server_model_param
        local_grad_vector = self.local_param_list

        model = self.model.to(self.device)
        loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer.zero_grad()

        examples = examples.to(self.device)
        labels = labels.to(self.device)
        ## Get esimated client loss
        loss_client = loss_function(model(examples), labels)

        # Get linear penalty on the current client parameters
        # Calculate the regularization term
        local_par_list = None

        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

        loss_penalty = torch.tensor(adaptive_alpha_coef * 0).to(self.device)

        if not local_grad_vector == 0:
            for avg_param, local_param in zip(server_model_param, local_grad_vector):
                loss_penalty = torch.tensor(adaptive_alpha_coef).to(
                    self.device
                ) * torch.sum(local_par_list * (-avg_param + local_param))
        loss_penalty = torch.mean(loss_penalty)
        loss = loss_client + loss_penalty
        loss.backward()

        self.optimizer.step()
        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_step_start(self, config, batch=None):
        super().train_step_start(config, batch)

        if not self.model_state_dict:
            self.model_state_dict = self.model.state_dict()

        server_model_param = []
        if self.model_state_dict:
            server_model_param = copy.deepcopy(self.model_state_dict)

        model_path = Config().params["model_path"]
        filename = f"{model_path}_{self.client_id}.pth"
        local_param_list = []
        if self.model_state_dict:
            local_param_list = 0
        if os.path.exists(filename):
            local_model = torch.load(filename)
            local_param_list = copy.deepcopy(local_model.state_dict())

        self.server_model_param = server_model_param
        self.local_param_list = local_param_list
