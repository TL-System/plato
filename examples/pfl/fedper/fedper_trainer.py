"""
A personalized federated learning trainer for FedPer.

"""
import logging

from bases import personalized_trainer
from bases import fedavg_partial


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model
    for normal and personalized learning processes."""

    def freeze_model(self, model, modules_name=None):
        """Freezing a part of the model."""
        if modules_name is not None:
            frozen_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = False
                    frozen_params.append(name)

            logging.info(
                "[Client #%d] has frozen %s",
                self.client_id,
                fedavg_partial.Algorithm.extract_modules_name(frozen_params),
            )

    def activate_model(self, model, modules_name=None):
        """Defreezing a part of the model."""
        if modules_name is not None:
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = True

    def train_run_start(self, config):
        """According to FedPer,
        1. freeze body of the model during personalization.
        """
        super().train_run_start(config)
        if self.personalized_learning:
            self.freeze_model(self.personalized_model, config["frozen_modules_name"])

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        if self.personalized_learning:
            self.activate_model(self.personalized_model, config["frozen_modules_name"])

        # assign the trained model to the personalized model during
        # the normal federated learning
        if not self.personalized_learning:
            self.copy_model_to_personalized_model(config)
