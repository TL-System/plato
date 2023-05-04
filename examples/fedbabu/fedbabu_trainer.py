"""
A personalized federated learning trainer using FedBABU.

"""
import logging
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the FedBABU algorithm."""

    def freeze_model(self, model, param_prefix=None):
        """Freeze a part of the model."""
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                logging.info("%s is freezed", name)
                param.requires_grad = False

    def activate_model(self, model, param_prefix=None):
        """Defreeze a part of the model."""
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = True

    def train_run_start(self, config):
        """According to FedBabu, freeze the classifier and
        never update it in federated learning phase."""
        self.freeze_model(self.model, config["classifier"])

    def train_run_end(self, config):
        """Activate the classifier."""
        self.activate_model(self.model, config["classifier"])
