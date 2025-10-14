"""
A self-supervised federated learning trainer with MoCoV2.

This trainer uses the composable trainer architecture with a custom callback
to implement MoCoV2-specific functionality (momentum updates).
"""

from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import self_supervised_learning as ssl_trainer


class MoCoV2Callback(TrainerCallback):
    """
    Callback implementing MoCoV2 algorithm functionality.

    Handles:
    - Momentum value computation using cosine scheduling
    - Model updates with Exponential Moving Average
    """

    def __init__(self):
        """Initialize MoCoV2-specific state."""
        # The momentum value used to update the model with Exponential Moving Average
        self.momentum_val = 0

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Compute the momentum value at the start of each epoch."""
        epoch = trainer.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        global_epoch = (trainer.current_round - 1) * config["epochs"] + epoch

        # Update the momentum value for the current epoch in regular federated training
        if not trainer.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(global_epoch, total_epochs, 0.996, 1)

    def on_train_step_start(self, trainer, config, batch, **kwargs):
        """Update the model based on the momentum value at each training step."""
        if not trainer.current_round > Config().trainer.rounds:
            # Update the model based on the momentum value
            # Specifically, it updates parameters of `encoder` with
            # Exponential Moving Average of `encoder_momentum`
            update_momentum(
                trainer.model.encoder,
                trainer.model.encoder_momentum,
                m=self.momentum_val,
            )
            update_momentum(
                trainer.model.projector,
                trainer.model.projector_momentum,
                m=self.momentum_val,
            )


class Trainer(ssl_trainer.Trainer):
    """
    A federated learning trainer using MoCoV2 algorithm.

    This trainer extends the SSL trainer with MoCoV2-specific functionality
    via a custom callback. It uses the composable trainer architecture with
    callbacks for momentum updates.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the MoCoV2 trainer with MoCoV2 callback.

        Arguments:
            model: The model to train (SSL model with momentum encoders)
            callbacks: List of callback classes or instances
        """
        # Create MoCoV2 callback
        mocov2_callback = MoCoV2Callback()

        # Combine with provided callbacks
        all_callbacks = [mocov2_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent SSL trainer with combined callbacks
        super().__init__(model=model, callbacks=all_callbacks)
