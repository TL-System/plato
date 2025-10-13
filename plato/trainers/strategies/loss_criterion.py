"""
Loss criterion strategy implementations.

This module provides default and common loss criterion strategies for
the composable trainer architecture.
"""

from typing import Optional

import torch
import torch.nn as nn

from plato.trainers import loss_criterion as loss_criterion_registry
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


class DefaultLossCriterionStrategy(LossCriterionStrategy):
    """
    Default loss criterion strategy using the framework's registry.

    This strategy uses the loss criterion from plato.trainers.loss_criterion
    registry, which is configured via the config file.

    Args:
        loss_fn: Optional custom loss function. If None, uses registry.

    Example:
        >>> strategy = DefaultLossCriterionStrategy()
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(self, loss_fn: Optional[callable] = None):
        """Initialize with optional custom loss function."""
        self.loss_fn = loss_fn
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize the loss criterion."""
        if self.loss_fn is None:
            # Use framework's registry
            self._criterion = loss_criterion_registry.get()
        else:
            self._criterion = self.loss_fn

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute loss using the configured criterion."""
        return self._criterion(outputs, labels)


class CrossEntropyLossStrategy(LossCriterionStrategy):
    """
    Cross-entropy loss strategy for classification tasks.

    Args:
        weight: Manual rescaling weight given to each class
        label_smoothing: Label smoothing factor (0.0 means no smoothing)
        reduction: Specifies the reduction to apply to the output
        ignore_index: Specifies a target value that is ignored

    Example:
        >>> strategy = CrossEntropyLossStrategy(label_smoothing=0.1)
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Initialize cross-entropy loss parameters."""
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize the cross-entropy loss criterion."""
        self._criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self._criterion(outputs, labels)


class MSELossStrategy(LossCriterionStrategy):
    """
    Mean squared error loss strategy for regression tasks.

    Args:
        reduction: Specifies the reduction to apply to the output

    Example:
        >>> strategy = MSELossStrategy()
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize MSE loss parameters."""
        self.reduction = reduction
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize the MSE loss criterion."""
        self._criterion = nn.MSELoss(reduction=self.reduction)

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return self._criterion(outputs, labels)


class BCEWithLogitsLossStrategy(LossCriterionStrategy):
    """
    Binary cross-entropy with logits loss for binary classification.

    Args:
        weight: Manual rescaling weight given to the loss
        reduction: Specifies the reduction to apply to the output
        pos_weight: Weight of positive examples

    Example:
        >>> strategy = BCEWithLogitsLossStrategy()
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Initialize BCE with logits loss parameters."""
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize the BCE with logits loss criterion."""
        self._criterion = nn.BCEWithLogitsLoss(
            weight=self.weight,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute BCE with logits loss."""
        return self._criterion(outputs, labels)


class NLLLossStrategy(LossCriterionStrategy):
    """
    Negative log likelihood loss strategy.

    Useful when combined with log-softmax activation.

    Args:
        weight: Manual rescaling weight given to each class
        reduction: Specifies the reduction to apply to the output
        ignore_index: Specifies a target value that is ignored

    Example:
        >>> strategy = NLLLossStrategy()
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Initialize NLL loss parameters."""
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize the NLL loss criterion."""
        self._criterion = nn.NLLLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute NLL loss."""
        return self._criterion(outputs, labels)


class CompositeLossStrategy(LossCriterionStrategy):
    """
    Composite loss strategy that combines multiple loss strategies.

    This allows combining different loss functions with weights,
    useful for multi-task learning or regularization.

    Args:
        strategies: List of (strategy, weight) tuples

    Example:
        >>> base_loss = CrossEntropyLossStrategy()
        >>> reg_loss = L2RegularizationStrategy(weight=0.01)
        >>> composite = CompositeLossStrategy([
        ...     (base_loss, 1.0),
        ...     (reg_loss, 0.1)
        ... ])
        >>> trainer = ComposableTrainer(loss_strategy=composite)
    """

    def __init__(self, strategies: list):
        """
        Initialize composite loss.

        Args:
            strategies: List of (strategy, weight) tuples or just strategies
                       If just strategies, all weights default to 1.0
        """
        self.strategies = []
        for item in strategies:
            if isinstance(item, tuple):
                strategy, weight = item
                self.strategies.append((strategy, weight))
            else:
                self.strategies.append((item, 1.0))

    def setup(self, context: TrainingContext) -> None:
        """Setup all component strategies."""
        for strategy, _ in self.strategies:
            strategy.setup(context)

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute weighted sum of all losses."""
        total_loss = torch.tensor(0.0, device=outputs.device)

        for strategy, weight in self.strategies:
            loss = strategy.compute_loss(outputs, labels, context)
            total_loss = total_loss + weight * loss

        return total_loss

    def teardown(self, context: TrainingContext) -> None:
        """Teardown all component strategies."""
        for strategy, _ in self.strategies:
            strategy.teardown(context)


class L2RegularizationStrategy(LossCriterionStrategy):
    """
    L2 regularization (weight decay) as a loss term.

    This can be composed with other losses to add explicit L2 regularization.
    Note: Most optimizers have built-in weight_decay, which is often preferred.

    Args:
        weight: Regularization weight (lambda)

    Example:
        >>> strategy = CompositeLossStrategy([
        ...     (CrossEntropyLossStrategy(), 1.0),
        ...     (L2RegularizationStrategy(weight=0.01), 1.0)
        ... ])
    """

    def __init__(self, weight: float = 0.01):
        """Initialize L2 regularization parameters."""
        self.weight = weight

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute L2 regularization term."""
        l2_loss = torch.tensor(0.0, device=outputs.device)

        for param in context.model.parameters():
            l2_loss = l2_loss + torch.sum(param**2)

        return self.weight * l2_loss
