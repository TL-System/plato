import torch

from plato.config import Config
from plato.trainers import basic

import copy
import os
import numpy as np


class Trainer(basic.Trainer):
    def perform_forward_and_backward_passes(
        self, config, examples, labels, alpha_coef, avg_mdl_param, local_grad_vector
    ):
        """Perform forward and backward passes in the training loop."""

        model = self.model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        self.optimizer.zero_grad()

        batch_x = examples.to(self.device)
        batch_y = labels.to(self.device)

        y_pred = model(batch_x)

        ## Get f_i estimate
        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
        loss_f_i = loss_f_i / list(batch_y.size())[0]

        # Get linear penalty on the current parameter estimates
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
        loss_algo = torch.tensor(alpha_coef * 0).to(loss_f_i.device)
        if not local_grad_vector == 0:
            for avg_param, local_param in zip(avg_mdl_param, local_grad_vector):
                loss_algo = torch.tensor(alpha_coef).to(loss_f_i.device) * torch.sum(
                    local_par_list * (-avg_param + local_param)
                )
        loss_algo = torch.mean(loss_algo)
        loss = loss_f_i + loss_algo
        loss.backward()
        self.optimizer.step()
        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]
        n_clnt = Config().clients.total_clients
        alpha_coef = Config().parameters.alpha_coef

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                examples, labels = examples.to(self.device), labels.to(self.device)

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

                clnt_y = labels.cpu().numpy()
                weight_list = clnt_y / np.sum(clnt_y) * n_clnt
                alpha_coef_adpt = alpha_coef / np.where(
                    weight_list != 0, weight_list, 1.0
                )

                loss = self.perform_forward_and_backward_passes(
                    config,
                    examples,
                    labels,
                    alpha_coef_adpt,
                    cld_mdl_param,
                    local_param_list,
                )

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                model_path = Config().params["model_path"]
                filename = f"{model_path}_{self.client_id}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)
