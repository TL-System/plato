"""
A federated learning trainer using Oort.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

import numpy as np
import torch
from torch import nn

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import LossCriterionStrategy


class OortLossStrategy(LossCriterionStrategy):
    """Loss strategy for Oort that tracks sum of squared per-sample losses."""

    def setup(self, context):
        """Initialize the loss criterion."""
        self._criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, outputs, labels, context):
        """
        Compute loss and track squared per-sample losses.

        This computes per-sample losses, tracks the sum of squares
        (used by Oort for client selection), and returns the mean loss.
        """
        per_sample_loss = self._criterion(outputs, labels)

        # Get the trainer from context to access run_history
        trainer = context.state.get("trainer")
        if trainer is not None:
            # Store the sum of squares over per_sample loss values
            trainer.run_history.update_metric(
                "train_squared_loss_step",
                sum(np.power(per_sample_loss.cpu().detach().numpy(), 2)),
            )

        return torch.mean(per_sample_loss)


class Trainer(ComposableTrainer):
    """A federated learning trainer for Oort that tracks squared losses."""

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Oort trainer.

        Args:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Create Oort-specific loss strategy
        loss_strategy = OortLossStrategy()

        # Initialize with Oort strategies
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
