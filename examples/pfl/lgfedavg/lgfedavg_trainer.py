"""
A personalized federated learning trainer using LG-FedAvg.

"""

import logging

from plato.utils import fonts
from plato.config import Config

from bases import personalized_trainer


class Trainer(personalized_trainer.Trainer):
    """A personalized federated learning trainer using the LG-FedAvg algorithm."""

    def preprocess_personalized_model(self, config):
        """In LG-FedAvg, only during the personalization process,
        the completed model will be assigned to the personalization model
        for direct evaluation."""
        if self.personalized_learning:
            self.copy_model_to_personalized_model(config)

    def freeze_model(self, model, modules_name=None):
        """Freeze a part of the model."""
        if modules_name is not None:
            frozen_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = False
                    frozen_params.append(name)

    def activate_model(self, model, modules_name=None):
        """Unfrozen a part of the model."""
        if modules_name is not None:
            unfrozen_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = True
                    unfrozen_params.append(name)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing one iteration of LG-FedAvg."""
        self.optimizer.zero_grad()

        outputs = self.forward_examples(examples)

        loss = self._loss_criterion(outputs, labels)

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        # first freeze the head and optimize the body
        self.freeze_model(self.model, config["head_modules_name"])
        self.activate_model(self.model, config["body_modules_name"])
        self.optimizer.step()

        # repeat the same optimization relying the optimized
        # body of the model
        self.optimizer.zero_grad()

        outputs = self.forward_examples(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        # first freeze the head and optimize the body
        self.freeze_model(self.model, config["body_modules_name"])
        self.activate_model(self.model, config["head_modules_name"])

        self.optimizer.step()

        return loss

    def train_run_end(self, config):
        """Copying the trained model to the personalized model."""

        # load the trained model to the personalized model
        self.copy_model_to_personalized_model()
