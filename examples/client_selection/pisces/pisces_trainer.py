"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

from plato.trainers import loss_criterion
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import LossCriterionStrategy


class PiscesLossStrategy(LossCriterionStrategy):
    """Loss strategy for Pisces that tracks per-batch loss values."""

    def setup(self, context):
        """Initialize the loss criterion."""
        self._criterion = loss_criterion.get()

    def compute_loss(self, outputs, labels, context):
        """
        Compute loss and track per-batch loss values.

        This computes the batch loss and stores it in run_history
        for Pisces client selection algorithm.
        """
        per_batch_loss = self._criterion(outputs, labels)

        # Get the trainer from context to access run_history
        trainer = context.state.get("trainer")
        current_epoch = getattr(context, "current_epoch", 1)
        if trainer is not None and current_epoch == 1:
            # Store scalar loss value for the first epoch only
            loss_value = float(per_batch_loss.detach().cpu().item())
            trainer.run_history.update_metric("train_batch_loss", loss_value)

        return per_batch_loss


class Trainer(ComposableTrainer):
    """The federated learning trainer for the Pisces client."""

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Pisces trainer.

        Args:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Create Pisces-specific loss strategy
        loss_strategy = PiscesLossStrategy()

        # Initialize with Pisces strategies
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=loss_strategy,
        )

    def train_model(self, config, trainset, sampler, **kwargs):
        """Training loop that provides trainer reference to context."""
        # Store trainer reference in context so loss strategy can access run_history
        self.context.state["trainer"] = self

        # Call parent training loop
        super().train_model(config, trainset, sampler, **kwargs)
