"""
A personalized federated learning trainer using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
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
            activated_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = True
                    activated_params.append(name)

            logging.info(
                "[Client #%d] has activated %s",
                self.client_id,
                fedavg_partial.Algorithm.extract_modules_name(activated_params),
            )

    def train_epoch_start(self, config):
        """
        Method called at the beginning of a training epoch.

        The local training stage in FedRep contains two parts:

        - Head optimization:
            Makes τ local gradient-based updates to solve for its optimal head given
            the current global representation communicated by the server.

        - Representation optimization:
            Takes one local gradient-based update with respect to the current representation.
        """
        # As presented in Section 3 of the FedRep paper, the head is optimized
        # for (epochs - 1) while freezing the representation.
        head_epochs = (
            config["head_epochs"] if "head_epochs" in config else config["epochs"] - 1
        )

        if self.current_epoch <= head_epochs:
            self.freeze_model(self.model, config["frozen_modules_name"])

        # The representation will then be optimized for only one epoch
        if self.current_epoch > head_epochs:
            self.activate_model(self.model, config["frozen_modules_name"])

    def personalized_train_run_start(self, config, **kwargs):
        """According to FedRep, freeze a partial of the model and
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
