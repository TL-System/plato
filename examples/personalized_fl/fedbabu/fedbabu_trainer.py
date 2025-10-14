"""
A personalized federated learning trainer with FedBABU.

This trainer uses the composable trainer architecture with a custom callback
to freeze/unfreeze layers during different training phases.
"""

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import basic
from plato.utils import trainer_utils


class FedBABUCallback(TrainerCallback):
    """
    Callback implementing FedBABU layer freezing functionality.

    FedBABU freezes the global model layers during the final personalization round,
    and freezes the local layers instead in regular rounds before the target number
    of rounds has been reached.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Freeze appropriate layers at the start of training."""
        if trainer.current_round > Config().trainer.rounds:
            # Personalization phase: freeze global layers
            trainer_utils.freeze_model(
                trainer.model,
                Config().algorithm.global_layer_names,
            )
        else:
            # Regular training phase: freeze local layers
            trainer_utils.freeze_model(
                trainer.model,
                Config().algorithm.local_layer_names,
            )

    def on_train_run_end(self, trainer, config, **kwargs):
        """Unfreeze layers at the end of training."""
        if trainer.current_round > Config().trainer.rounds:
            # Personalization phase: unfreeze global layers
            trainer_utils.activate_model(
                trainer.model, Config().algorithm.global_layer_names
            )
        else:
            # Regular training phase: unfreeze local layers
            trainer_utils.activate_model(
                trainer.model, Config().algorithm.local_layer_names
            )


class Trainer(basic.Trainer):
    """
    A federated learning trainer using FedBABU algorithm.

    This trainer freezes the global model layers in the final personalization round,
    and freezes the local layers instead in regular rounds before the target number
    of rounds has been reached.

    It uses the composable trainer architecture with a callback for layer freezing.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedBABU trainer with layer freezing callback.

        Arguments:
            model: The model to train
            callbacks: List of callback classes or instances
        """
        # Create FedBABU callback
        fedbabu_callback = FedBABUCallback()

        # Combine with provided callbacks
        all_callbacks = [fedbabu_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent trainer with combined callbacks
        super().__init__(model=model, callbacks=all_callbacks)
