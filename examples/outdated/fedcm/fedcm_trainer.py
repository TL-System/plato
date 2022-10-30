import torch
import torch.nn as nn
import numpy as np
import logging
import os

from plato.config import Config
from plato.trainers import basic


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype("float32")
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx : idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)
        self.n_par = len(get_mdl_params([self.model])[0])
        # init delta
        self.delta = None

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """A custom training loop."""
        max_norm = 10
        log_interval = 10
        alpha = config["alpha"]
        lr = Config().parameters.optimizer.lr * (
            Config().algorithm.lr_decay ** self.current_round
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        batch_size = config["batch_size"]
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

        num_epochs = config["epochs"]
        lr_schedule = None
        if "lr_schedule" in config:
            lr_schedule = optimizers.get_lr_schedule(
                optimizer, len(train_loader), train_loader
            )

        if self.delta:
            self.delta = get_mdl_params([self.delta], self.n_par)[0]
        else:
            self.delta = np.zeros(self.n_par).astype("float32")

        self.delta = torch.tensor(self.delta, dtype=torch.float32, device=self.device)
        self.delta = self.delta / (len(train_loader) * num_epochs * lr)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, num_epochs + 1):
            batch_id = 0
            for examples, labels in train_loader:
                # examples = examples.view(len(examples), -1)
                examples, labels = examples.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                logits = self.model(examples)
                loss = criterion(logits, labels)

                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat(
                            (local_par_list, param.reshape(-1)), 0
                        )

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(),
                            epoch,
                            num_epochs,
                            batch_id,
                            len(train_loader),
                            loss.data.item(),
                        )
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id,
                            epoch,
                            num_epochs,
                            batch_id,
                            len(train_loader),
                            loss.data.item(),
                        )

                loss_algo = torch.sum(local_par_list * self.delta)
                loss = alpha * loss + (alpha - 1) * loss_algo

                # print("train loss: ", loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_norm
                )  # Clip gradients to prevent exploding
                optimizer.step()
                batch_id = batch_id + 1
            lr_scheduler.step()
