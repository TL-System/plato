"""
Customized Trainer for PerFedRLNAS using the composable trainer architecture.

This trainer uses custom strategies for optimizer and loss criterion specific to NASVIT.
"""

import fednasvit_specific
import torch

from plato.config import Config
from plato.trainers import basic
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    OptimizerStrategy,
    TrainingContext,
)
from plato.trainers.strategies.lr_scheduler import TimmLRSchedulerStrategy

# ============================================================================
# Custom Strategies
# ============================================================================


class NASVITLossCriterionStrategy(LossCriterionStrategy):
    """
    Loss criterion strategy for NASVIT.

    Uses the special loss criterion defined in fednasvit_specific module.
    """

    def __init__(self):
        """Initialize the NASVIT loss strategy."""
        self._loss_criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize NASVIT loss criterion."""
        self._loss_criterion = fednasvit_specific.get_nasvit_loss_criterion()

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute loss using NASVIT-specific criterion."""
        return self._loss_criterion(outputs, labels)


class NASVITOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy for NASVIT.

    Uses the special optimizer defined in fednasvit_specific module.
    """

    def create_optimizer(
        self, model, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create NASVIT-specific optimizer."""
        return fednasvit_specific.get_optimizer(model)


# ============================================================================
# Trainer Class
# ============================================================================


class Trainer(ComposableTrainer):
    """
    Use special optimizer and loss criterion specific for NASVIT.

    This trainer extends ComposableTrainer with NASVIT-specific functionality
    via custom strategies for optimizer and loss criterion. It uses the
    composable trainer architecture with strategy injection.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the NASVIT trainer with custom strategies.

        Arguments:
            model: The model to train
            callbacks: List of callback classes or instances
        """
        # Create NASVIT-specific strategies
        loss_strategy = NASVITLossCriterionStrategy()
        optimizer_strategy = NASVITOptimizerStrategy()

        # Determine if we need timm scheduler
        lr_scheduler_strategy = None
        if Config().trainer.lr_scheduler == "timm":
            lr_scheduler_strategy = TimmLRSchedulerStrategy()

        # Initialize with strategies
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=loss_strategy,
            optimizer_strategy=optimizer_strategy,
            lr_scheduler_strategy=lr_scheduler_strategy,
        )
