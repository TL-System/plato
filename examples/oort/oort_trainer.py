"""The training loop that takes place on clients of Oort."""

from collections import OrderedDict

import math
import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def __init__(self, model=None):
        super().__init__(model)

        self.loss_dict = {}

    def train_run_start(self, config):
        """
        Method called at the start of training run.
        """
        self.loss_dict = OrderedDict()

    def train_step_end(self, config, batch, loss):
        """
        Method called at the end of a training step.

        :param batch: the current batch of training data.
        :param loss: the loss computed in the current batch.
        """
        # Track the loss
        if self.current_epoch == 1:
            self.loss_dict[batch] = math.pow(float(loss), 2)
        else:
            self.loss_dict[batch] += math.pow(float(loss), 2)

    def train_run_end(self, config):
        """
        Method called at the end of training run.
        """
        model_name = Config().trainer.model_name
        model_path = Config().params["checkpoint_path"]
        sum_loss = 0
        for batch_id in self.loss_dict:
            sum_loss += self.loss_dict[batch_id]
        sum_loss /= config["epochs"]
        filename = f"{model_path}/{model_name}_{self.client_id}_squared_batch_loss.pth"
        torch.save(sum_loss, filename)
