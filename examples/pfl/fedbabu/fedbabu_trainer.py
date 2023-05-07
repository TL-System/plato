"""
A personalized federated learning trainer for FedBABU.

"""
import logging

from bases import personalized_trainer
from bases import fedavg_partial


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model."""

    def freeze_model(self, model, modules_name=None):
        """Freeze a part of the model."""
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
        """Defreeze a part of the model."""
        if modules_name is not None:
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = True

    def train_run_start(self, config):
        """According to FedBABU, freeze a partial of the model and
        never update it in federated learning phase.
        """
        super().train_run_start(config)
        if self.personalized_learning:
            self.freeze_model(self.personalized_model, config["frozen_modules_name"])
        else:
            self.freeze_model(self.model, config["frozen_modules_name"])

    def train_run_end(self, config):
        """Activate the model."""
        super().train_run_end(config)
        if self.personalized_learning:
            self.activate_model(self.personalized_model, config["frozen_modules_name"])
        else:
            self.activate_model(self.model, config["frozen_modules_name"])
