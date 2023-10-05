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
        self.cld_mdl_param = None
        self.local_param_list = None

    # pylint:disable=too-many-locals
    def perform_forward_and_backward_passes(
        self,
        config,
        examples,
        labels,
    ):
        """Perform forward and backward passes in the training loop."""
        clnt_y = labels.cpu().numpy()
        weight_list = clnt_y / np.sum(clnt_y) * Config().clients.total_clients
        alpha_coef_adpt = Config().parameters.alpha_coef / np.where(
            weight_list != 0, weight_list, 1.0
        )
        # According to original source code, they use cld_mdl_param_tensor
        #   as avg_mdl_param in the train_feddyn_mdl function
        avg_mdl_param = self.cld_mdl_param
        local_grad_vector = self.local_param_list

        model = self.model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        self.optimizer.zero_grad()

        examples = examples.to(self.device)
        labels = labels.to(self.device)
        ## Get f_i estimate
        loss_f_i = loss_fn(model(examples), labels.reshape(-1).long())
        loss_f_i = loss_f_i / list(labels.size())[0]

        # Get linear penalty on the current parameter estimates
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
        loss_algo = torch.tensor(alpha_coef_adpt * 0).to(loss_f_i.device)
        if not local_grad_vector == 0:
            for avg_param, local_param in zip(avg_mdl_param, local_grad_vector):
                loss_algo = torch.tensor(alpha_coef_adpt).to(
                    loss_f_i.device
                ) * torch.sum(local_par_list * (-avg_param + local_param))
        loss_algo = torch.mean(loss_algo)
        loss = loss_f_i + loss_algo
        loss.backward()
        self.optimizer.step()
        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_step_start(self, config, batch=None):
        super().train_step_start(config, batch)

        if not self.model_state_dict:
            self.model_state_dict = self.model.state_dict()
        cld_mdl_param = []
        if self.model_state_dict:
            cld_mdl_param = copy.deepcopy(self.model_state_dict)

        model_path = Config().params["model_path"]
        filename = f"{model_path}_{self.client_id}.pth"
        local_param_list = []
        if self.model_state_dict:
            local_param_list = 0
        if os.path.exists(filename):
            local_model = torch.load(filename)
            local_param_list = copy.deepcopy(local_model.state_dict())

        self.cld_mdl_param = cld_mdl_param
        self.local_param_list = local_param_list
