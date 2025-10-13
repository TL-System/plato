"""
Optimizer strategy implementations.

This module provides default and common optimizer strategies for
the composable trainer architecture.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import optimizers as optimizer_registry
from plato.trainers.strategies.base import OptimizerStrategy, TrainingContext


class DefaultOptimizerStrategy(OptimizerStrategy):
    """
    Default optimizer strategy using the framework's registry.

    This strategy uses the optimizer from plato.trainers.optimizers
    registry, which is configured via the config file.

    Args:
        optimizer_fn: Optional custom optimizer factory. If None, uses registry.

    Example:
        >>> strategy = DefaultOptimizerStrategy()
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(self, optimizer_fn: Optional[callable] = None):
        """Initialize with optional custom optimizer factory."""
        self.optimizer_fn = optimizer_fn

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create optimizer using registry or custom function."""
        if self.optimizer_fn is None:
            # Use framework's registry
            return optimizer_registry.get(model)
        else:
            return self.optimizer_fn(model)


class SGDOptimizerStrategy(OptimizerStrategy):
    """
    Stochastic Gradient Descent optimizer strategy.

    Args:
        lr: Learning rate (if None, uses config)
        momentum: Momentum factor
        weight_decay: Weight decay (L2 penalty)
        dampening: Dampening for momentum
        nesterov: Whether to use Nesterov momentum

    Example:
        >>> strategy = SGDOptimizerStrategy(lr=0.01, momentum=0.9)
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        lr: Optional[float] = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        """Initialize SGD optimizer parameters."""
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def setup(self, context: TrainingContext) -> None:
        """Get learning rate from config if not specified."""
        if self.lr is None:
            self.lr = Config().trainer.lr

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create SGD optimizer."""
        return torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov,
        )


class AdamOptimizerStrategy(OptimizerStrategy):
    """
    Adam optimizer strategy.

    Args:
        lr: Learning rate (if None, uses config)
        betas: Coefficients for computing running averages
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay (L2 penalty)
        amsgrad: Whether to use AMSGrad variant

    Example:
        >>> strategy = AdamOptimizerStrategy(lr=0.001, betas=(0.9, 0.999))
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        lr: Optional[float] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """Initialize Adam optimizer parameters."""
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def setup(self, context: TrainingContext) -> None:
        """Get learning rate from config if not specified."""
        if self.lr is None:
            self.lr = Config().trainer.lr

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create Adam optimizer."""
        return torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class AdamWOptimizerStrategy(OptimizerStrategy):
    """
    AdamW optimizer strategy with decoupled weight decay.

    AdamW is often preferred over Adam for better regularization.

    Args:
        lr: Learning rate (if None, uses config)
        betas: Coefficients for computing running averages
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay (L2 penalty)
        amsgrad: Whether to use AMSGrad variant

    Example:
        >>> strategy = AdamWOptimizerStrategy(lr=0.001, weight_decay=0.01)
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        lr: Optional[float] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        """Initialize AdamW optimizer parameters."""
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def setup(self, context: TrainingContext) -> None:
        """Get learning rate from config if not specified."""
        if self.lr is None:
            self.lr = Config().trainer.lr

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class RMSpropOptimizerStrategy(OptimizerStrategy):
    """
    RMSprop optimizer strategy.

    Args:
        lr: Learning rate (if None, uses config)
        alpha: Smoothing constant
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay (L2 penalty)
        momentum: Momentum factor
        centered: If True, compute centered RMSprop

    Example:
        >>> strategy = RMSpropOptimizerStrategy(lr=0.01, alpha=0.99)
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        lr: Optional[float] = None,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        """Initialize RMSprop optimizer parameters."""
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

    def setup(self, context: TrainingContext) -> None:
        """Get learning rate from config if not specified."""
        if self.lr is None:
            self.lr = Config().trainer.lr

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create RMSprop optimizer."""
        return torch.optim.RMSprop(
            model.parameters(),
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )


class ParameterGroupOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy with custom parameter groups.

    This allows different learning rates or settings for different
    parts of the model (e.g., different rates for backbone vs head).

    Args:
        optimizer_class: Optimizer class to use (e.g., torch.optim.Adam)
        parameter_groups_fn: Function that takes model and returns list of param groups
        default_lr: Default learning rate (if None, uses config)
        optimizer_kwargs: Additional kwargs for optimizer

    Example:
        >>> def create_param_groups(model):
        ...     return [
        ...         {'params': model.backbone.parameters(), 'lr': 0.001},
        ...         {'params': model.head.parameters(), 'lr': 0.01}
        ...     ]
        >>> strategy = ParameterGroupOptimizerStrategy(
        ...     optimizer_class=torch.optim.Adam,
        ...     parameter_groups_fn=create_param_groups
        ... )
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        optimizer_class: type,
        parameter_groups_fn: callable,
        default_lr: Optional[float] = None,
        **optimizer_kwargs,
    ):
        """Initialize parameter group optimizer strategy."""
        self.optimizer_class = optimizer_class
        self.parameter_groups_fn = parameter_groups_fn
        self.default_lr = default_lr
        self.optimizer_kwargs = optimizer_kwargs

    def setup(self, context: TrainingContext) -> None:
        """Get learning rate from config if not specified."""
        if self.default_lr is None:
            self.default_lr = Config().trainer.lr

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create optimizer with custom parameter groups."""
        param_groups = self.parameter_groups_fn(model)

        # Add default lr to groups that don't specify it
        for group in param_groups:
            if "lr" not in group:
                group["lr"] = self.default_lr

        return self.optimizer_class(param_groups, **self.optimizer_kwargs)


class GradientClippingOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy wrapper that adds gradient clipping.

    This wraps another optimizer strategy and applies gradient clipping
    after each backward pass.

    Args:
        base_strategy: The underlying optimizer strategy
        max_norm: Max norm of gradients
        norm_type: Type of norm to use (2 for L2 norm)

    Example:
        >>> base = AdamOptimizerStrategy(lr=0.001)
        >>> strategy = GradientClippingOptimizerStrategy(
        ...     base_strategy=base,
        ...     max_norm=1.0
        ... )
        >>> trainer = ComposableTrainer(optimizer_strategy=strategy)
    """

    def __init__(
        self,
        base_strategy: OptimizerStrategy,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ):
        """Initialize gradient clipping optimizer strategy."""
        self.base_strategy = base_strategy
        self.max_norm = max_norm
        self.norm_type = norm_type

    def setup(self, context: TrainingContext) -> None:
        """Setup base strategy."""
        self.base_strategy.setup(context)

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create optimizer using base strategy."""
        return self.base_strategy.create_optimizer(model, context)

    def on_optimizer_step(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> None:
        """Apply gradient clipping before optimizer step."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            context.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

        # Call base strategy's hook if it exists
        self.base_strategy.on_optimizer_step(optimizer, context)

    def teardown(self, context: TrainingContext) -> None:
        """Teardown base strategy."""
        self.base_strategy.teardown(context)
