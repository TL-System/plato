"""
A trainer with FedRep.
"""
from plato.trainers import basic
from plato.config import Config

from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """A trainer with FedRep."""

    def train_run_start(self, config):
        """Freeze the global layers during the final personalization round."""
        super().train_run_start(config)

        if self.current_round > Config().trainer.rounds:
            # Freeze the model body while only optimizing the local layers
            # furing the final personalization
            trainer_utils.freeze_model(
                self.model, Config().algorithm.global_layer_names
            )

            # Set the number of epochs for personalization
            if hasattr(Config().algorithm.personalization, "epochs"):
                config["epochs"] = Config().algorithm.personalization.epochs

    def train_epoch_start(self, config):
        """
        Local training in FedRep contains two phases:

        - Optimizing the local layers:
            Go through a number of epoches involving local gradient-based updates,
            with the current shared global layers frozen.

        - Optimizing the global layers:
            Using the remaining number of epoches to optimize the global layers.
        """
        super().train_epoch_start(config)

        if self.current_round <= Config().trainer.rounds:
            # As presented in Section 3 of the paper, the local layers is
            # optimized for a certain number of epochs while freezing the global
            # layers
            local_epochs = (
                Config().algorithm.local_epochs
                if hasattr(Config().algorithm, "local_epochs")
                else config["epochs"] - 1
            )

            if self.current_epoch <= local_epochs:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.global_layer_names,
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.local_layer_names
                )
            # The global layers will then be optimized for the remaining epochs
            else:
                trainer_utils.freeze_model(
                    self.model,
                    Config().algorithm.local_layer_names,
                )
                trainer_utils.activate_model(
                    self.model, Config().algorithm.global_layer_names
                )
        else:
            # The global layers in the model will be frozen during the final
            # personalization round
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_layer_names,
            )

    def train_run_end(self, config):
        """Activate the model."""
        super().train_run_end(config)

        if self.current_round > Config().trainer.rounds:
            trainer_utils.activate_model(
                self.model, Config().algorithm.global_layer_names
            )
