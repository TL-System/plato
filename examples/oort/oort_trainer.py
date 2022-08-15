"""The training loop that takes place on clients of Oort."""


import numpy as np
import torch
from torch import nn

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        # For getting the training loss of each sample
        return nn.CrossEntropyLoss(reduction="none")

    def process_loss(self, loss):
        """Process loss with any additional steps."""
        sample_loss = loss.cpu().detach().numpy()
        self.run_history.update_metric(
            "train_squared_loss_step", sum(np.power(sample_loss, 2))
        )

        # Return the mean of leach sample's loss for backward
        return torch.mean(loss)

    def train_run_end(self, config):
        """
        Method called at the end of training run.
        """
        train_squared_loss_step = self.run_history.get_metric_values(
            "train_squared_loss_step"
        )

        self.run_history.update_metric(
            "train_squared_loss_sum",
            sum(train_squared_loss_step)
            / (Config().data.partition_size * config["epochs"]),
        )
