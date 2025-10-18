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

    def __init__(self):
        self._criterion = None
        self._run_history = None

    def setup(self, context):
        """Initialize the loss criterion."""
        self._criterion = nn.CrossEntropyLoss(reduction="none")

    def attach_run_history(self, run_history):
        """Attach run history for metric tracking."""
        self._run_history = run_history

    def compute_loss(self, outputs, labels, context):
        """
        Compute loss and track squared per-sample losses.

        This computes per-sample losses, tracks the sum of squares
        (used by Oort for client selection), and returns the mean loss.
        """
        per_sample_loss = self._criterion(outputs, labels)

        if self._run_history is not None:
            # Store the sum of squares over per_sample loss values
            self._run_history.update_metric(
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

        if hasattr(self.loss_strategy, "attach_run_history"):
            self.loss_strategy.attach_run_history(self.run_history)
