"""
A personalized federated learning trainer using LG-FedAvg.
"""


from pflbases import trainer_utils
from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the LG-FedAvg algorithm."""

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing one iteration of LG-FedAvg."""

        # LG-FedAvg will first only train local layers
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.activate_model(self.model, Config().algorithm.local_layer_names)

        super().perform_forward_and_backward_passes(config, examples, labels)

        # Then, LG-FedAvg will only train non-local layers
        trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.freeze_model(
            self.model,
            Config().algorithm.local_layer_names,
        )

        loss = super().perform_forward_and_backward_passes(config, examples, labels)

        return loss
