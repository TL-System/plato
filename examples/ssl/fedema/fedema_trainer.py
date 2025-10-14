"""
A self-supervised federated learning trainer with FedEMA.

This trainer uses the composable trainer architecture with custom callbacks
and loss strategy to implement FedEMA-specific functionality (dual loss computation
and momentum updates).
"""

import torch
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import loss_criterion
from plato.trainers import self_supervised_learning as ssl_trainer
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


class FedEMALossCriterionStrategy(LossCriterionStrategy):
    """
    Loss criterion strategy for FedEMA with dual loss computation.

    FedEMA computes the loss as the average of two SSL losses when
    the model outputs are tuples/lists (dual views).
    """

    def __init__(self):
        """Initialize the FedEMA loss strategy."""
        self._ssl_criterion = None
        self._personalization_criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize loss criterion."""
        self._ssl_criterion = loss_criterion.get()

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute FedEMA loss based on current training phase."""
        current_round = context.current_round

        # Personalization phase - use standard SSL loss
        if current_round > Config().trainer.rounds:
            if self._personalization_criterion is None:
                loss_criterion_type = Config().algorithm.personalization.loss_criterion
                loss_criterion_params = {}
                if hasattr(Config().parameters.personalization, "loss_criterion"):
                    loss_criterion_params = (
                        Config().parameters.personalization.loss_criterion._asdict()
                    )
                self._personalization_criterion = loss_criterion.get(
                    loss_criterion=loss_criterion_type,
                    loss_criterion_params=loss_criterion_params,
                )
            return self._personalization_criterion(outputs, labels)

        # SSL training phase - use FedEMA dual loss
        else:
            if isinstance(outputs, (list, tuple)):
                # FedEMA: average of two losses
                loss = 0.5 * (
                    self._ssl_criterion(*outputs[0]) + self._ssl_criterion(*outputs[1])
                )
                return loss
            else:
                return self._ssl_criterion(outputs)


class FedEMACallback(TrainerCallback):
    """
    Callback implementing FedEMA algorithm functionality.

    Handles:
    - Momentum value computation using cosine scheduling
    - Model updates with Exponential Moving Average
    """

    def __init__(self):
        """Initialize FedEMA-specific state."""
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
            # Exponential Moving Average of `momentum_encoder`
            update_momentum(
                trainer.model.encoder,
                trainer.model.momentum_encoder,
                m=self.momentum_val,
            )
            update_momentum(
                trainer.model.projector,
                trainer.model.momentum_projector,
                m=self.momentum_val,
            )


class Trainer(ssl_trainer.Trainer):
    """
    A federated learning trainer using FedEMA algorithm.

    This trainer extends the SSL trainer with FedEMA-specific functionality
    via a custom callback and loss strategy. It uses the composable trainer
    architecture with callbacks for momentum updates and a custom loss strategy
    for dual loss computation.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedEMA trainer with FedEMA callback and loss strategy.

        Arguments:
            model: The model to train (SSL model with momentum encoders)
            callbacks: List of callback classes or instances
        """
        # Create FedEMA callback
        fedema_callback = FedEMACallback()

        # Combine with provided callbacks
        all_callbacks = [fedema_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent SSL trainer - we'll override the loss strategy
        super().__init__(model=model, callbacks=all_callbacks)

        # Replace the SSL loss strategy with FedEMA loss strategy
        self.loss_strategy = FedEMALossCriterionStrategy()
        self.loss_strategy.setup(self.context)
