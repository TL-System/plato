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
        """According to FedBabu, freeze a partial of the model and
        never update it in federated learning phase."""
        self.freeze_model(self.model, config["frozen_modules_name"])

    def train_run_end(self, config):
        """Activate the model."""
        self.activate_model(self.model, config["frozen_modules_name"])

    def personalized_train_run_start(self, config, **kwargs):
        """According to FedBabu, freeze a partial of the model and
        never update it during personalization."""
        eval_outputs = super().personalized_train_run_start(config, **kwargs)
        logging.info(
            "[Client #%d] will freeze %s before performing personalization",
            self.client_id,
            config["frozen_personalized_modules_name"],
        )
        self.freeze_model(
            self.personalized_model, config["frozen_personalized_modules_name"]
        )
        return eval_outputs

    def personalized_train_run_end(self, config):
        """Reactive the personalized model."""
        self.activate_model(
            self.personalized_model, config["frozen_personalized_modules_name"]
        )
