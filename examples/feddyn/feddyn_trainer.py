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

from plato.config import Config
from plato.trainers import basic


# pylint:disable=no-member
class Trainer(basic.Trainer):
    """
    FedDyn's Trainer.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.global_model_param = None
        self.local_param_last_epoch = None

    def perform_forward_and_backward_passes(
        self,
        config,
        examples,
        labels,
    ):
        """Perform forward and backward passes in the training loop."""
        weight_list = labels / torch.sum(labels) * Config().clients.total_clients

        alpha_coef = (
            Config().algorithm.alpha_coef
            if hasattr(Config().algorithm, "alpha_coef")
            else 0.01
        )
        adaptive_alpha_coef = alpha_coef / torch.where(
            weight_list != 0, weight_list, 1.0
        )

        self.optimizer.zero_grad()
        outputs = self.model(examples)

        # In the paper's formulation (1), the loss has three parts.
        # (1) The ordinary loss such as CrossEntropy.
        # (2) The linear penalty, we need to calculate the dot product between the
        # current model parameters and model updates in the previous round.
        # (3) The L2 loss, which is realized by the weight decay in the optimizer.

        # Get oridinary loss of the task
        loss_task = self._loss_criterion(outputs, labels)

        # Get linear penalty on the current client parameters.
        local_params = self.model.state_dict()
        loss_penalty = torch.zeros(adaptive_alpha_coef.shape).to(self.device)
        adaptive_alpha_coef = torch.Tensor(adaptive_alpha_coef).to(self.device)

        for parameter_name in local_params:
            loss_penalty += adaptive_alpha_coef * torch.sum(
                local_params[parameter_name]
                * (
                    -self.global_model_param[parameter_name].to(self.device)
                    + self.local_param_last_epoch[parameter_name].to(self.device)
                )
            )

        loss_penalty = torch.sum(loss_penalty)
        loss = loss_task + loss_penalty
        loss.backward()

        self.optimizer.step()
        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_run_start(self, config):
        super().train_run_start(config)
        # At the beginning of each round, the client model parameters are the same as the
        # global model parameters.
        self.global_model_param = copy.deepcopy(self.model.state_dict())

        model_path = Config().params["model_path"]
        filename = f"{model_path}_{self.client_id}.pth"
        if os.path.exists(filename):
            self.local_param_last_epoch = torch.load(filename).state_dict()
        else:
            # If it does not exist, this client has not trained any model yet.
            # The client model parameters in last epoch are the same as the global model
            # parameters.
            self.local_param_last_epoch = copy.deepcopy(self.model.state_dict())
