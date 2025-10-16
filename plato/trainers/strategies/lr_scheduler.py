"""
Learning rate scheduler strategy implementations.

This module provides default and common LR scheduler strategies for
the composable trainer architecture.
"""

from typing import Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import lr_schedulers as lr_scheduler_registry
from plato.trainers.strategies.base import LRSchedulerStrategy, TrainingContext


class DefaultLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Default LR scheduler strategy using the framework's registry.

    This strategy uses the LR scheduler from plato.trainers.lr_schedulers
    registry, which is configured via the config file.

    Args:
        scheduler_fn: Optional custom scheduler factory. If None, uses registry.

    Example:
        >>> strategy = DefaultLRSchedulerStrategy()
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, scheduler_fn: Optional[callable] = None):
        """Initialize with optional custom scheduler factory."""
        self.scheduler_fn = scheduler_fn

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create scheduler using registry or custom function."""
        if self.scheduler_fn is None:
            # Use framework's registry
            # Check if scheduler is configured
            if "lr_scheduler" not in context.config:
                return None
            train_loader = context.state.get("train_loader")
            if train_loader is None:
                iterations_per_epoch = 0
            else:
                try:
                    iterations_per_epoch = len(train_loader)
                except TypeError:
                    iterations_per_epoch = 0

            if iterations_per_epoch <= 0:
                iterations_per_epoch = 1

            return lr_scheduler_registry.get(optimizer, iterations_per_epoch)
        else:
            return self.scheduler_fn(optimizer)


class NoSchedulerStrategy(LRSchedulerStrategy):
    """
    No learning rate scheduling - keeps LR constant.

    Example:
        >>> strategy = NoSchedulerStrategy()
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Return None for no scheduling."""
        return None


class StepLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Step learning rate scheduler - decays LR by gamma every step_size epochs.

    Args:
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch

    Example:
        >>> strategy = StepLRSchedulerStrategy(step_size=10, gamma=0.1)
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, step_size: int = 10, gamma: float = 0.1, last_epoch: int = -1):
        """Initialize step LR scheduler parameters."""
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create step LR scheduler."""
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
        )


class MultiStepLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Multi-step learning rate scheduler - decays LR at specific milestones.

    Args:
        milestones: List of epoch indices at which to decay LR
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch

    Example:
        >>> strategy = MultiStepLRSchedulerStrategy(
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, milestones: list, gamma: float = 0.1, last_epoch: int = -1):
        """Initialize multi-step LR scheduler parameters."""
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create multi-step LR scheduler."""
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
        )


class ExponentialLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Exponential learning rate scheduler - decays LR exponentially.

    Args:
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch

    Example:
        >>> strategy = ExponentialLRSchedulerStrategy(gamma=0.95)
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, gamma: float = 0.95, last_epoch: int = -1):
        """Initialize exponential LR scheduler parameters."""
        self.gamma = gamma
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create exponential LR scheduler."""
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.gamma, last_epoch=self.last_epoch
        )


class CosineAnnealingLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Cosine annealing learning rate scheduler.

    Args:
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch

    Example:
        >>> strategy = CosineAnnealingLRSchedulerStrategy(T_max=50)
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        """Initialize cosine annealing LR scheduler parameters."""
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create cosine annealing LR scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )


class CosineAnnealingWarmRestartsSchedulerStrategy(LRSchedulerStrategy):
    """
    Cosine annealing with warm restarts (SGDR) learning rate scheduler.

    Args:
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase T_i after each restart
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch

    Example:
        >>> strategy = CosineAnnealingWarmRestartsSchedulerStrategy(
        ...     T_0=10,
        ...     T_mult=2
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(
        self,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize cosine annealing warm restarts scheduler parameters."""
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create cosine annealing warm restarts scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )


class ReduceLROnPlateauSchedulerStrategy(LRSchedulerStrategy):
    """
    Reduce learning rate when a metric has stopped improving.

    Note: This scheduler requires manual stepping with validation metrics.

    Args:
        mode: 'min' or 'max' - whether lower or higher metric is better
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement to wait
        threshold: Threshold for measuring new optimum
        threshold_mode: 'rel' or 'abs'
        cooldown: Number of epochs to wait before resuming normal operation
        min_lr: Minimum learning rate
        eps: Minimal decay applied to lr

    Example:
        >>> strategy = ReduceLROnPlateauSchedulerStrategy(
        ...     mode='min',
        ...     factor=0.1,
        ...     patience=10
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(
        self,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ):
        """Initialize reduce on plateau scheduler parameters."""
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create reduce on plateau scheduler."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
        )

    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext,
    ) -> None:
        """
        Step with validation metric.

        Note: For ReduceLROnPlateau, you need to provide a metric.
        This implementation uses the last train loss from context.
        For validation loss, override this method.
        """
        if scheduler is not None:
            # Get metric from context (default to last train loss)
            metric = context.state.get("last_loss", 0.0)
            scheduler.step(metric)


class LinearLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Linear learning rate scheduler - linearly changes LR.

    Args:
        start_factor: Multiplicative factor at the start
        end_factor: Multiplicative factor at the end
        total_iters: Number of iterations to reach end_factor
        last_epoch: The index of last epoch

    Example:
        >>> strategy = LinearLRSchedulerStrategy(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(
        self,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        """Initialize linear LR scheduler parameters."""
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create linear LR scheduler."""
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.start_factor,
            end_factor=self.end_factor,
            total_iters=self.total_iters,
            last_epoch=self.last_epoch,
        )


class PolynomialLRSchedulerStrategy(LRSchedulerStrategy):
    """
    Polynomial learning rate scheduler.

    Args:
        total_iters: Number of iterations to reach minimum learning rate
        power: Power of the polynomial
        last_epoch: The index of last epoch

    Example:
        >>> strategy = PolynomialLRSchedulerStrategy(
        ...     total_iters=50,
        ...     power=2.0
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(self, total_iters: int = 5, power: float = 1.0, last_epoch: int = -1):
        """Initialize polynomial LR scheduler parameters."""
        self.total_iters = total_iters
        self.power = power
        self.last_epoch = last_epoch

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create polynomial LR scheduler."""
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=self.total_iters,
            power=self.power,
            last_epoch=self.last_epoch,
        )


class WarmupSchedulerStrategy(LRSchedulerStrategy):
    """
    Learning rate warmup followed by another scheduler.

    Args:
        warmup_epochs: Number of epochs for warmup
        warmup_start_lr: Starting learning rate for warmup
        base_scheduler: The scheduler to use after warmup

    Example:
        >>> base = CosineAnnealingLRSchedulerStrategy(T_max=50)
        >>> strategy = WarmupSchedulerStrategy(
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0.0001,
        ...     base_scheduler=base
        ... )
        >>> trainer = ComposableTrainer(lr_scheduler_strategy=strategy)
    """

    def __init__(
        self,
        warmup_epochs: int,
        warmup_start_lr: float,
        base_scheduler: Optional[LRSchedulerStrategy] = None,
    ):
        """Initialize warmup scheduler parameters."""
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_scheduler = base_scheduler
        self._warmup_scheduler = None
        self._base_scheduler_obj = None
        self._current_epoch = 0

    def setup(self, context: TrainingContext) -> None:
        """Setup base scheduler if provided."""
        if self.base_scheduler is not None:
            self.base_scheduler.setup(context)

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create warmup scheduler.

        Note: This returns the warmup scheduler, but tracks the base scheduler internally.
        """
        # Get initial LR from optimizer
        initial_lr = optimizer.param_groups[0]["lr"]

        # Calculate warmup factor
        start_factor = self.warmup_start_lr / initial_lr if initial_lr > 0 else 1.0

        # Create warmup scheduler
        self._warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Create base scheduler if provided
        if self.base_scheduler is not None:
            self._base_scheduler_obj = self.base_scheduler.create_scheduler(
                optimizer, context
            )

        self._current_epoch = 0
        return self._warmup_scheduler

    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext,
    ) -> None:
        """Step the appropriate scheduler based on current epoch."""
        self._current_epoch += 1

        if self._current_epoch <= self.warmup_epochs:
            # During warmup
            if self._warmup_scheduler is not None:
                self._warmup_scheduler.step()
        else:
            # After warmup, use base scheduler
            if self._base_scheduler_obj is not None:
                if self.base_scheduler is not None:
                    self.base_scheduler.step(self._base_scheduler_obj, context)
                else:
                    self._base_scheduler_obj.step()

    def teardown(self, context: TrainingContext) -> None:
        """Teardown base scheduler if provided."""
        if self.base_scheduler is not None:
            self.base_scheduler.teardown(context)


class TimmLRSchedulerStrategy(LRSchedulerStrategy):
    """
    LR Scheduler strategy for timm (PyTorch Image Models) schedulers.

    This strategy handles timm schedulers that require step_update() calls
    during training steps, in addition to epoch-level step() calls. This is
    necessary for schedulers like CosineLRScheduler that update learning rate
    per iteration rather than per epoch.

    The strategy tracks the number of updates (training steps) and calls
    step_update() after each batch, while also calling step() at the end
    of each epoch.

    Args:
        None

    Example:
        >>> strategy = TimmLRSchedulerStrategy()
        >>> trainer = TrainerWithTimmScheduler(
        ...     lr_scheduler_strategy=strategy
        ... )

    Note:
        This strategy requires the LR scheduler to be configured in the
        config file. It uses the plato.trainers.lr_schedulers registry
        to create the scheduler.
    """

    def __init__(self):
        """Initialize timm scheduler strategy."""
        super().__init__()
        self.num_updates = 0
        self.past_epochs = 0

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create timm scheduler using configuration.

        Args:
            optimizer: The optimizer to schedule
            context: Training context

        Returns:
            LR scheduler instance from timm, or None if not configured
        """
        # Import locally to avoid dependency if not used
        from plato.trainers import lr_schedulers

        train_loader = context.state.get("train_loader")
        if train_loader is None:
            return None

        config = context.config
        if "lr_scheduler" not in config:
            return None

        scheduler = lr_schedulers.get(optimizer, len(train_loader))

        # Initialize for global lr scheduler if needed
        if config.get("global_lr_scheduler", False):
            past_epochs = (context.current_round - 1) * config.get("epochs", 1)
            self.past_epochs = past_epochs
            if scheduler is not None:
                scheduler.step(past_epochs)
                scheduler.step_update(past_epochs * len(train_loader))

        return scheduler

    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext,
    ) -> None:
        """
        Perform epoch-level scheduler step for timm.

        Args:
            scheduler: The timm scheduler
            context: Training context
        """
        if scheduler is not None:
            config = context.config
            if config.get("global_lr_scheduler", False):
                scheduler.step(self.past_epochs + context.current_epoch + 1)
            else:
                scheduler.step(context.current_epoch + 1)

    def on_epoch_start(self, scheduler, context: TrainingContext) -> None:
        """
        Called at epoch start to initialize num_updates.

        This method should be called at the beginning of each epoch
        to properly track the number of training steps.

        Args:
            scheduler: The scheduler (unused in this method)
            context: Training context
        """
        train_loader = context.state.get("train_loader")
        if train_loader is None:
            return

        self.num_updates = context.current_epoch * len(train_loader)

        config = context.config
        if config.get("global_lr_scheduler", False):
            self.num_updates += self.past_epochs * len(train_loader)

    def on_step(self, scheduler, context: TrainingContext) -> None:
        """
        Called after each training step to update timm scheduler.

        This method should be called after each optimizer.step() to
        update the learning rate on a per-iteration basis.

        Args:
            scheduler: The timm scheduler
            context: Training context
        """
        self.num_updates += 1
        if scheduler is not None:
            scheduler.step_update(num_updates=self.num_updates)
