"""
A self-supervised federated learning trainer with SimSiam.

This trainer uses the composable trainer architecture with a custom loss strategy
to implement SimSiam-specific functionality (dual loss computation).
"""

import torch

from plato.config import Config
from plato.trainers import loss_criterion
from plato.trainers import self_supervised_learning as ssl_trainer
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


class SimSiamLossCriterionStrategy(LossCriterionStrategy):
    """
    Loss criterion strategy for SimSiam with dual loss computation.

    SimSiam computes the loss as the average of two SSL losses when
    the model outputs are tuples/lists (dual views).
    """

    def __init__(self):
        """Initialize the SimSiam loss strategy."""
        self._ssl_criterion = None
        self._personalization_criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize loss criterion."""
        self._ssl_criterion = loss_criterion.get()

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute SimSiam loss based on current training phase."""
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

        # SSL training phase - use SimSiam dual loss
        else:
            if isinstance(outputs, (list, tuple)):
                # SimSiam: average of two losses
                loss = 0.5 * (
                    self._ssl_criterion(*outputs[0]) + self._ssl_criterion(*outputs[1])
                )
                return loss
            else:
                return self._ssl_criterion(outputs)


class Trainer(ssl_trainer.Trainer):
    """
    A federated learning trainer using SimSiam algorithm.

    This trainer extends the SSL trainer with SimSiam-specific functionality
    via a custom loss strategy. It uses the composable trainer architecture
    with a custom loss strategy for dual loss computation.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the SimSiam trainer with SimSiam loss strategy.

        Arguments:
            model: The model to train (SSL model)
            callbacks: List of callback classes or instances
        """
        # Initialize parent SSL trainer
        super().__init__(model=model, callbacks=callbacks)

        # Replace the SSL loss strategy with SimSiam loss strategy
        self.loss_strategy = SimSiamLossCriterionStrategy()
        self.loss_strategy.setup(self.context)
