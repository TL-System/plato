"""
A personalized federated learning trainer using FedBABU.

"""
import logging
from plato.trainers import basic_personalized
from plato.algorithms import fedavg_partial


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the FedBABU algorithm."""

    def freeze_model(self, model, modules_name=None):
        """Freeze a part of the model."""
        if modules_name is not None:
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = False

            frozen_params = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad is False
            ]
            logging.info(
                "[Client #%d] has frozen %s during normal federated training",
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
        """According to FedBabu, freeze a partial of the model and
        never update it in federated learning phase."""
        self.freeze_model(self.model, config["frozen_modules_name"])

    def train_run_end(self, config):
        """Activate the model."""
        self.activate_model(self.model, config["frozen_modules_name"])
