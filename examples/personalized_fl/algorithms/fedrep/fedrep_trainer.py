"""
A trainer for FedRep approach.
"""

from pflbases import trainer_utils

from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """A trainer for FedRep."""

    def train_run_start(self, config):
        """Freeze the body during personalization."""
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            # Freeze the model body while only optimizing the head
            # furing the final personalization
            trainer_utils.freeze_model(
                self.model, Config().algorithm.global_module_names
            )
            # Set the number of epochs for personalization
            config["epochs"] = Config().algorithm.personalization.epochs

    def train_epoch_start(self, config):
        """A training epoch for FedRep.
        The local training stage in FedRep contains two parts:

        - Head optimization:
            Makes τ local gradient-based updates to solve for its optimal head given
            the current global representation communicated by the server.

        - Representation optimization:
            Takes one local gradient-based update with respect to the current representation.
        """
        super().train_epoch_start(config)

        if self.current_round <= Config().trainer.rounds:
            # As presented in Section 3 of the FedRep paper, the head is optimized
            # for (epochs - 1) while freezing the representation.
            head_epochs = (
                Config().algorithm.head_epochs
                if hasattr(Config().algorithm, "head_epochs")
                else config["epochs"] - 1
            )

            if self.current_epoch <= head_epochs:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.global_module_names,
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.local_module_names
                )

            # The representation will then be optimized for only one epoch
            if self.current_epoch > head_epochs:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.local_module_names,
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.global_module_names
                )
        else:
            # The body of the model will be frozen during the
            # final personalization
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_module_names,
            )

    def train_run_end(self, config):
        """Activating the model."""
        super().train_run_end(config)
        trainer_utils.activate_model(self.model, Config().algorithm.global_module_names)
