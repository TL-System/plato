"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

import copy
import logging
import pickle
from collections import OrderedDict

import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        self.server_control_variate = None
        self.client_control_variate = None

        # Save the global model weights for computing new control variate
        # using the Option 2 in the paper
        self.global_model_weights = None

        # Path to the client control variate
        self.client_control_variate_path = None

        self.additional_data = None
        self.param_groups = None
        self.local_steps = 0
        self.local_lr = None
        self.client_control_variate_delta = None

    def get_optimizer(self, model):
        """Gets the parameter groups from the optimizer"""
        optimizer = super().get_optimizer(model)
        self.param_groups = optimizer.param_groups
        # SCAFFOLD requires η (local learning rate)
        if len(self.param_groups) > 0 and "lr" in self.param_groups[0]:
            self.local_lr = self.param_groups[0]["lr"]
        return optimizer

    def train_run_start(self, config):
        """Initializes the client control variate to 0 if the client
        is participating for the first time, and reset local counters.
        """
        self.server_control_variate = self.additional_data
        if self.client_control_variate is None:
            self.client_control_variate = {}
            for variate in self.server_control_variate:
                self.client_control_variate[variate] = torch.zeros(
                    self.server_control_variate[variate].shape
                )
        self.global_model_weights = copy.deepcopy(self.model.state_dict())
        self.local_steps = 0
        self.client_control_variate_delta = OrderedDict(
            (k, torch.zeros_like(v)) for k, v in self.server_control_variate.items()
        )

    def train_step_end(self, config, batch=None, loss=None):
        """Modifies the weights based on the server and client control variates,
        and count local steps for SCAFFOLD scaling.
        """
        for group in self.param_groups:
            learning_rate = group["lr"]
            counter = 0
            for name in self.server_control_variate:
                if "weight" in name or "bias" in name:
                    server_control_variate = self.server_control_variate[name].to(
                        self.device
                    )
                    param = group["params"][counter]
                    if self.client_control_variate is not None:
                        param.data.add_(
                            torch.sub(
                                server_control_variate,
                                self.client_control_variate[name].to(self.device),
                            ),
                            alpha=learning_rate,
                        )
                    else:
                        param.data.add_(server_control_variate, alpha=learning_rate)
                    counter += 1
        self.local_steps += 1

    def train_run_end(self, config):
        """Compute Δc_i per SCAFFOLD (Eq. 4/5) and update c_i.

        c_i' = c - (1/(η·τ)) (x_local - x_global)
        Δc_i = c_i' - c_i
        c_i ← c_i'
        """
        eta = self.local_lr if self.local_lr is not None else Config().trainer.lr
        tau = max(1, int(self.local_steps))

        delta_ci = OrderedDict()
        new_client_control_variate = OrderedDict()

        for name, x_global in self.global_model_weights.items():
            x_local = self.model.state_dict()[name]
            ci_old = self.client_control_variate[name]

            ci_new = self.server_control_variate[name].to(self.device) - (
                x_local.to(self.device) - x_global.to(self.device)
            ) / (eta * tau)

            delta = ci_new - ci_old.to(self.device)
            delta_ci[name] = delta.detach().cpu()
            new_client_control_variate[name] = ci_new.detach().cpu()

        # Expose Δc_i for the outbound processor and update stored c_i
        self.client_control_variate_delta = delta_ci
        self.client_control_variate = new_client_control_variate

        # Save client control variate
        logging.info(
            "[Client #%d] Saving the control variate to %s.",
            self.client_id,
            self.client_control_variate_path,
        )
        with open(self.client_control_variate_path, "wb") as path:
            pickle.dump(self.client_control_variate, path)

        logging.info(
            "[Client #%d] Control variate saved to %s.",
            self.client_id,
            self.client_control_variate_path,
        )
