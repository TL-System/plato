"""
A personalized federated learning trainer with LG-FedAvg.
"""
from plato.trainers import basic
from plato.config import Config
from plato.utils import trainer_utils


class Trainer(basic.Trainer):
    """
    The training loop in LG-FedAvg performs two forward and backward passes in
    one iteration. It first freezes the global model layers and trains the local
    layers, and then freezes the local layers and trains global layers to finish
    one training loop.
    """

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing one iteration of LG-FedAvg."""

        # LG-FedAvg first only trains local layers
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.activate_model(self.model, Config().algorithm.local_layer_names)

        super().perform_forward_and_backward_passes(config, examples, labels)

        # Secondly, LG-FedAvg only trains non-local layers
        trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.freeze_model(self.model, Config().algorithm.local_layer_names)

        loss = super().perform_forward_and_backward_passes(config, examples, labels)

        return loss
