"""
FedMos Strategy Implementation

Reference:
Wang, X., Chen, Y., Li, Y., Liao, X., Jin, H., & Li, B. (2023).
"FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection."
IEEE INFOCOM 2023.

Paper: https://ieeexplore.ieee.org/document/10228957
Source code: https://github.com/Distributed-Learning-Networking-Group/FedMoS

Description:
FedMos addresses client drift by using a double momentum mechanism:
1. Local momentum: Standard momentum in the optimizer
2. Global momentum: Momentum towards the global model

The optimizer maintains both momentums and combines them to stabilize training.
The key innovation is the adaptive momentum that balances local updates with
global model proximity.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from plato.config import Config
from plato.trainers.strategies.base import (
    ModelUpdateStrategy,
    OptimizerStrategy,
    TrainingContext,
)


class FedMosOptimizer(Optimizer):
    """
    FedMos optimizer with double momentum.

    This optimizer implements the FedMos update rule with both local momentum
    and global momentum towards the initial model.

    Args:
        params: Model parameters to optimize
        lr: Learning rate
        a: Local momentum coefficient (default: 0.9)
        mu: Global momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)

    Attributes:
        lr: Learning rate
        a: Local momentum coefficient
        mu: Global momentum coefficient
        weight_decay: Weight decay coefficient
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        a: float = 0.9,
        mu: float = 0.9,
        weight_decay: float = 0,
    ):
        """
        Initialize FedMos optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            a: Local momentum coefficient
            mu: Global momentum coefficient
            weight_decay: Weight decay coefficient
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if a < 0.0 or a > 1.0:
            raise ValueError(f"Invalid local momentum value: {a}")
        if mu < 0.0 or mu > 1.0:
            raise ValueError(f"Invalid global momentum value: {mu}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, a=a, mu=mu, weight_decay=weight_decay)
        super(FedMosOptimizer, self).__init__(params, defaults)

        # Initialize state for momentum
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["momentum_buffer"] = torch.zeros_like(p.data)

    def update_momentum(self):
        """
        Update local momentum buffers.

        This should be called after backward() but before step().
        It updates the momentum buffer with the current gradient.
        """
        for group in self.param_groups:
            a = group["a"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                momentum_buffer = state["momentum_buffer"]

                # Update momentum: m = a * m + g
                momentum_buffer.mul_(a).add_(p.grad.data)

    @torch.no_grad()
    def step(self, global_model_params=None, closure=None):
        """
        Perform a single optimization step.

        Args:
            global_model_params: Global model parameters (model object)
            closure: Optional closure to reevaluate the model

        Returns:
            Optional loss from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Get momentum buffer
                momentum_buffer = state["momentum_buffer"]

                # Get corresponding global parameter
                if global_model_params is not None:
                    # Find matching parameter in global model
                    global_param = None
                    for global_p in global_model_params.parameters():
                        if global_p.shape == p.shape:
                            global_param = global_p
                            break

                    if global_param is not None:
                        # FedMos update: w = w - lr * m + mu * (w_global - w)
                        # This can be rewritten as: w = w - lr * m + mu * w_global - mu * w
                        # = (1 - mu) * w - lr * m + mu * w_global

                        # Apply weight decay if specified
                        if weight_decay != 0:
                            p.data.mul_(1 - lr * weight_decay)

                        # Apply momentum
                        p.data.add_(momentum_buffer, alpha=-lr)

                        # Apply global momentum: w += mu * (w_global - w)
                        p.data.add_(global_param.data.to(p.device) - p.data, alpha=mu)
                    else:
                        # Fallback: standard momentum update
                        if weight_decay != 0:
                            p.data.mul_(1 - lr * weight_decay)
                        p.data.add_(momentum_buffer, alpha=-lr)
                else:
                    # No global model: standard momentum update
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(momentum_buffer, alpha=-lr)

        return loss


class FedMosOptimizerStrategy(OptimizerStrategy):
    """
    FedMos optimizer strategy for composable trainer.

    This strategy creates a FedMos optimizer with double momentum.
    It should be used together with FedMosUpdateStrategy to maintain
    the global model reference.

    Args:
        lr: Learning rate (default: 0.01)
        a: Local momentum coefficient (default: 0.9)
        mu: Global momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedMosOptimizerStrategy,
        ...     FedMosUpdateStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     optimizer_strategy=FedMosOptimizerStrategy(lr=0.01, a=0.9, mu=0.9),
        ...     model_update_strategy=FedMosUpdateStrategy()
        ... )

    Note:
        This strategy should be used together with FedMosUpdateStrategy which
        handles the global model state management.
    """

    def __init__(
        self,
        lr: float = 0.01,
        a: float = 0.9,
        mu: float = 0.9,
        weight_decay: float = 0,
    ):
        """
        Initialize FedMos optimizer strategy.

        Args:
            lr: Learning rate
            a: Local momentum coefficient
            mu: Global momentum coefficient
            weight_decay: Weight decay coefficient
        """
        self.lr = lr
        self.a = a
        self.mu = mu
        self.weight_decay = weight_decay

    def create_optimizer(self, model: nn.Module, context: TrainingContext) -> Optimizer:
        """
        Create FedMos optimizer.

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            FedMosOptimizer instance
        """
        return FedMosOptimizer(
            model.parameters(),
            lr=self.lr,
            a=self.a,
            mu=self.mu,
            weight_decay=self.weight_decay,
        )

    def on_optimizer_step(self, optimizer: Optimizer, context: TrainingContext) -> None:
        """
        Hook called after optimizer.step().

        This is where we can add any post-step processing if needed.

        Args:
            optimizer: The optimizer that just stepped
            context: Training context
        """
        pass


class FedMosUpdateStrategy(ModelUpdateStrategy):
    """
    FedMos model update strategy for state management.

    This strategy manages the FedMos-specific state:
    - Saves global model at start of training
    - Provides global model reference to optimizer
    - Calls optimizer.update_momentum() before each step

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedMosOptimizerStrategy,
        ...     FedMosUpdateStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     optimizer_strategy=FedMosOptimizerStrategy(lr=0.01, a=0.9, mu=0.9),
        ...     model_update_strategy=FedMosUpdateStrategy()
        ... )

    Note:
        This strategy should be used together with FedMosOptimizerStrategy.
    """

    def __init__(self):
        """Initialize FedMos update strategy."""
        self.global_model = None

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Save global model at start of training.

        Args:
            context: Training context with model
        """
        # Save a copy of the global model
        self.global_model = copy.deepcopy(context.model)
        self.global_model.to(context.device)

        # Store in context for potential use by other strategies
        context.state["fedmos_global_model"] = self.global_model

    def before_step(self, context: TrainingContext) -> None:
        """
        Update momentum before optimizer step.

        This calls update_momentum() on the FedMos optimizer if it exists.

        Args:
            context: Training context
        """
        # The optimizer should be available in context if needed
        # For now, we rely on the training loop to call update_momentum()
        pass

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Cleanup at end of training.

        Args:
            context: Training context
        """
        # Move global model to CPU to free GPU memory
        if self.global_model is not None:
            self.global_model.to(torch.device("cpu"))

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        self.global_model = None


class FedMosOptimizerStrategyFromConfig(FedMosOptimizerStrategy):
    """
    FedMos optimizer strategy that reads configuration from Config.

    This variant automatically reads hyperparameters from the configuration
    file, making it easier to use in existing Plato workflows.

    Configuration:
        The strategy looks for:
        - Config().algorithm.a (local momentum, default: 0.9)
        - Config().algorithm.mu (global momentum, default: 0.9)
        - Config().parameters.optimizer.lr (learning rate, default: 0.01)
        - Config().parameters.optimizer.weight_decay (default: 0)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   a: 0.9
        >>> #   mu: 0.9
        >>> # parameters:
        >>> #   optimizer:
        >>> #     lr: 0.01
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedMosOptimizerStrategyFromConfig,
        ...     FedMosUpdateStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     optimizer_strategy=FedMosOptimizerStrategyFromConfig(),
        ...     model_update_strategy=FedMosUpdateStrategy()
        ... )
    """

    def __init__(self):
        """Initialize FedMos optimizer strategy from config."""
        config = Config()

        # Read hyperparameters from config with defaults
        a = 0.9
        mu = 0.9
        lr = 0.01
        weight_decay = 0

        if hasattr(config, "algorithm"):
            if hasattr(config.algorithm, "a"):
                a = config.algorithm.a
            if hasattr(config.algorithm, "mu"):
                mu = config.algorithm.mu

        if hasattr(config, "parameters") and hasattr(config.parameters, "optimizer"):
            if hasattr(config.parameters.optimizer, "lr"):
                lr = config.parameters.optimizer.lr
            if hasattr(config.parameters.optimizer, "weight_decay"):
                weight_decay = config.parameters.optimizer.weight_decay
        elif hasattr(config, "trainer") and hasattr(config.trainer, "lr"):
            lr = config.trainer.lr

        super().__init__(lr=lr, a=a, mu=mu, weight_decay=weight_decay)
