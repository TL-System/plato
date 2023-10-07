"""
A personalized federated learning trainer for FedPer approach.
"""

from pflbases import personalized_trainer
from pflbases import trainer_utils

from plato.config import Config


class Trainer(personalized_trainer.Trainer):
    """A trainer to freeze and activate modules of one model
    for normal and personalized learning processes."""

    def train_run_start(self, config):
        """Freezing the body"""
        super().train_run_start(config)
        if self.personalized_learning:
            trainer_utils.freeze_model(
                self.personalized_model,
                Config().algorithm.global_modules_name,
                log_info=f"[Client #{self.client_id}]",
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
        if not self.personalized_learning:
            # As presented in Section 3 of the FedRep paper, the head is optimized
            # for (epochs - 1) while freezing the representation.
            head_epochs = (
                config["head_epochs"]
                if "head_epochs" in config
                else config["epochs"] - 1
            )

            if self.current_epoch <= head_epochs:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.global_modules_name,
                    log_info=f"[Client #{self.client_id}]",
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.head_modules_name
                )

            # The representation will then be optimized for only one epoch
            if self.current_epoch > head_epochs:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.head_modules_name,
                    log_info=f"[Client #{self.client_id}]",
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.global_modules_name
                )

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        if self.personalized_learning:
            trainer_utils.activate_model(
                self.personalized_model, Config().algorithm.global_modules_name
            )
