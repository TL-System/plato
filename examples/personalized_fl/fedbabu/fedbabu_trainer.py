"""
A personalized federated learning trainer with FedBABU.
"""
from plato.config import Config
from plato.trainers import basic
from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """
    A trainer with FedBABU, which freezes the global model layers in the final
    personalization round, and freezes the local layers instead in the regular
    rounds before the target number of rounds has been reached.
    """

    def train_run_start(self, config):
        super().train_run_start(config)
        if self.current_round > Config().trainer.rounds:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.global_layer_names,
            )
        else:
            trainer_utils.freeze_model(
                self.model,
                Config().algorithm.local_layer_names,
            )

    def train_run_end(self, config):
        super().train_run_end(config)

        if self.current_round > Config().trainer.rounds:
            trainer_utils.activate_model(
                self.model, Config().algorithm.global_layer_names
            )
        else:
            trainer_utils.activate_model(
                self.model, Config().algorithm.local_layer_names
            )
