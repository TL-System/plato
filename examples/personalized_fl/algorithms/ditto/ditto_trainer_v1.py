"""
A personalized federated learning trainer using Ditto.

This implementation striclty follows the official code presented in
https://github.com/litian96/ditto.

"""

import torch
from torch.nn.functional import cross_entropy

from pflbases import personalized_trainer


class Trainer(personalized_trainer.Trainer):
    """A trainer of Ditto approach to optimize the global model and the
    personalized model in a sequence manner."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the lambda used in the Ditto paper
        self.ditto_lambda = 0.0

        self.personalized_optimizer = None

    def models_norm_distance(self, norm=2):
        """Compute the distance between the personalized model and the
        global model."""
        size = 0
        for param in self.personalized_model.parameters():
            if param.requires_grad:
                size += param.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        for param, global_param in zip(
            self.personalized_model.parameters(),
            self.model.parameters(),
        ):
            if param.requires_grad and global_param.requires_grad:
                sum_var[size : size + param.view(-1).shape[0]] = (
                    (param - global_param)
                ).view(-1)
                size += param.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def get_loss_criterion(self):
        """Get the loss of Ditto approach."""

        def ditto_loss(outputs, labels):
            return (
                cross_entropy(outputs, labels)
                + self.ditto_lambda * self.models_norm_distance()
            )

        return ditto_loss

    def preprocess_personalized_model(self, config):
        """Do nothing to the loaded personalized model in APFL."""

    def train_run_start(self, config):
        """Define personalization before running."""
        super().train_run_start(config)

        # initialize the optimizer, lr_schedule, and loss criterion
        self.personalized_optimizer = self.get_personalized_optimizer()

        self.personalized_model.to(self.device)
        self.personalized_model.train()

        # initialize the lambda
        self.ditto_lambda = config["ditto_lambda"]

    def train_epoch_start(self, config):
        """Assigning the lr of optimizer to the personalized optimizer."""
        super().train_epoch_start(config)
        self.personalized_optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[
            0
        ]["lr"]

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.

        """
        # optimize the personalized model
        self.personalized_optimizer.zero_grad()
        outputs = self.personalized_model(examples)
        personalized_loss = self._loss_criterion(outputs, labels)
        personalized_loss.backward()
        self.personalized_optimizer.step()

        # perform normal local update
        super().perform_forward_and_backward_passes(config, examples, labels)

        return personalized_loss
