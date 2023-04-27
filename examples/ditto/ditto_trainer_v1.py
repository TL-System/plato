"""
A personalized federated learning trainer using Ditto.

This implementation striclty follows the official code presented in
https://github.com/litian96/ditto.

"""

import os

import torch
from torch.nn.functional import cross_entropy

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the lambda used in the Ditto paper
        self.ditto_lambda = 0.0

    def models_norm_distance(self, norm=2):
        """Compute the distance between the personalized model and the
        global model."""
        size = 0
        for param in self.personalized_model.parameters():
            if param.requires_grad:
                size += param.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        for (param, global_param) in zip(
            self.personalized_model.parameters(),
            self.model.parameters(),
        ):
            if param.requires_grad and global_param.requires_grad:
                sum_var[size : size + param.view(-1).shape[0]] = (
                    (param - global_param)
                ).view(-1)
                size += param.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def get_personalized_loss_criterion(self):
        """Get the loss of Ditto approach."""

        def ditto_loss(outputs, labels):

            return (
                cross_entropy(outputs, labels)
                + self.ditto_lambda * self.models_norm_distance()
            )

        return ditto_loss

    def train_run_start(self, config):
        """Define personalization before running."""

        # initialize the optimizer, lr_schedule, and loss criterion
        self.personalized_optimizer = self.get_personalized_optimizer(
            self.personalized_model
        )
        self.personalized_optimizer = self._adjust_lr(
            config, self.lr_scheduler, self.personalized_optimizer
        )
        self._personalized_loss_criterion = self.get_personalized_loss_criterion()

        self.personalized_model.to(self.device)
        self.personalized_model.train()

        # initialize the lambda
        self.ditto_lambda = config["ditto_lambda"]

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
        personalized_loss = self._personalized_loss_criterion(outputs, labels)
        personalized_loss.backward()
        self.personalized_optimizer.step()

        # perform normal local update
        super().perform_forward_and_backward_passes(config, examples, labels)

        return personalized_loss

    def train_run_end(self, config):
        """Saving the personalized model and lambda."""
        # save the personalized model for current round
        # to the model dir of this client
        if "max_concurrency" in config:

            current_round = self.current_round

            learning_dict = {"lambda": self.ditto_lambda}
            personalized_model_name = config["personalized_model_name"]
            save_location = self.get_checkpoint_dir_path()
            filename = NameFormatter.get_format_name(
                client_id=self.client_id,
                model_name=personalized_model_name,
                round_n=current_round,
                run_id=None,
                prefix="personalized",
                ext="pth",
            )
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(
                filename=filename, location=save_location, learning_dict=learning_dict
            )

    def personalized_train_model(self, config, trainset, sampler, **kwargs):
        """Ditto will only evaluate the personalized model."""
        batch_size = config["personalized_batch_size"]

        testset = kwargs["testset"]
        testset_sampler = kwargs["testset_sampler"]

        personalized_test_loader = self.get_personalized_data_loader(
            batch_size, testset, testset_sampler.get()
        )

        eval_outputs = self.perform_evaluation(
            personalized_test_loader, self.personalized_model
        )
        accuracy = eval_outputs["accuracy"]

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy, current_round=self.current_round, epoch=0, run_id=None
        )

        if "max_concurrency" in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config["personalized_model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
            return None
