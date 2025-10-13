"""
LG-FedAvg Strategy Implementation

Reference:
Liang, P. P., Liu, T., Ziyin, L., Allen, N. B., Auerbach, R. P., Brent, D., ... & Morency, L. P. (2020).
"Think Locally, Act Globally: Federated Learning with Local and Global Representations."
arXiv preprint arXiv:2001.01523.

Paper: https://arxiv.org/abs/2001.01523

Description:
LG-FedAvg (Local and Global FedAvg) is a personalized federated learning approach
that divides the model into two parts:
- Global layers: Shared across all clients and aggregated on the server
- Local layers: Kept locally on each client and not shared

During training, LG-FedAvg performs two forward/backward passes per iteration:
1. First pass: Train only local layers (with global layers frozen)
2. Second pass: Train only global layers (with local layers frozen)

This allows clients to learn personalized representations while still benefiting
from shared global features.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers.strategies.base import TrainingContext, TrainingStepStrategy


class LGFedAvgStepStrategy(TrainingStepStrategy):
    """
    LG-FedAvg training step strategy with dual forward/backward passes.

    This strategy implements the LG-FedAvg training procedure which trains
    local and global layers separately in each training step. This enables
    personalized federated learning where local layers capture client-specific
    patterns while global layers learn shared representations.

    The training procedure for each step:
    1. Freeze global layers, activate local layers
    2. Perform forward pass, compute loss, backward pass, optimizer step
    3. Freeze local layers, activate global layers
    4. Perform forward pass, compute loss, backward pass, optimizer step
    5. Re-activate all layers

    Args:
        global_layer_names: List of layer name patterns for global layers.
                           Layers matching these patterns will be shared.
        local_layer_names: List of layer name patterns for local layers.
                          Layers matching these patterns will be kept local.
        train_local_first: If True, trains local layers first (default).
                          If False, trains global layers first.

    Attributes:
        global_layer_names: Patterns for identifying global layers
        local_layer_names: Patterns for identifying local layers
        train_local_first: Training order flag

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import LGFedAvgStepStrategy
        >>>
        >>> # For a model with 'fc1', 'fc2' (global) and 'fc3' (local)
        >>> strategy = LGFedAvgStepStrategy(
        ...     global_layer_names=['fc1', 'fc2'],
        ...     local_layer_names=['fc3']
        ... )
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
        >>>
        >>> # From config
        >>> strategy = LGFedAvgStepStrategyFromConfig()

    Note:
        The layer names should match the names used in model.named_parameters().
        Partial matching is supported (e.g., 'fc' will match 'fc.weight' and 'fc.bias').
    """

    def __init__(
        self,
        global_layer_names: List[str],
        local_layer_names: List[str],
        train_local_first: bool = True,
    ):
        """
        Initialize LG-FedAvg training step strategy.

        Args:
            global_layer_names: List of layer name patterns for global layers
            local_layer_names: List of layer name patterns for local layers
            train_local_first: If True, train local layers first
        """
        if not global_layer_names:
            raise ValueError("global_layer_names cannot be empty")
        if not local_layer_names:
            raise ValueError("local_layer_names cannot be empty")

        self.global_layer_names = global_layer_names
        self.local_layer_names = local_layer_names
        self.train_local_first = train_local_first

    def _set_requires_grad(
        self, model: nn.Module, layer_names: List[str], requires_grad: bool
    ) -> None:
        """
        Enable or disable gradients for specific layers.

        Args:
            model: The neural network model
            layer_names: List of layer name patterns to match
            requires_grad: Whether to enable (True) or disable (False) gradients

        Note:
            This method uses partial matching. A parameter is affected if any
            pattern in layer_names appears in its name.
        """
        for name, param in model.named_parameters():
            # Check if this parameter belongs to any of the specified layers
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = requires_grad

    def _freeze_layers(self, model: nn.Module, layer_names: List[str]) -> None:
        """
        Freeze specific layers (disable gradients).

        Args:
            model: The neural network model
            layer_names: List of layer name patterns to freeze
        """
        self._set_requires_grad(model, layer_names, False)

    def _activate_layers(self, model: nn.Module, layer_names: List[str]) -> None:
        """
        Activate specific layers (enable gradients).

        Args:
            model: The neural network model
            layer_names: List of layer name patterns to activate
        """
        self._set_requires_grad(model, layer_names, True)

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """
        Perform LG-FedAvg training step with dual forward/backward passes.

        This method trains local and global layers separately:
        1. First pass: Train one set of layers with the other frozen
        2. Second pass: Train the other set with the first frozen
        3. Re-enable all gradients

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch (already on device)
            labels: Target labels (already on device)
            loss_criterion: Callable that computes loss
            context: Training context

        Returns:
            Loss value from the second pass (for logging)

        Note:
            The loss returned is from the second training pass, which is
            typically the global layer training. Both passes contribute
            to model updates.
        """
        # Determine training order
        if self.train_local_first:
            first_layers = self.local_layer_names
            first_freeze = self.global_layer_names
            second_layers = self.global_layer_names
            second_freeze = self.local_layer_names
        else:
            first_layers = self.global_layer_names
            first_freeze = self.local_layer_names
            second_layers = self.local_layer_names
            second_freeze = self.global_layer_names

        # First pass: Train first set of layers
        self._freeze_layers(model, first_freeze)
        self._activate_layers(model, first_layers)

        optimizer.zero_grad()
        outputs = model(examples)
        loss_first = loss_criterion(outputs, labels)
        loss_first.backward()
        optimizer.step()

        # Second pass: Train second set of layers
        self._freeze_layers(model, second_freeze)
        self._activate_layers(model, second_layers)

        optimizer.zero_grad()
        outputs = model(examples)
        loss_second = loss_criterion(outputs, labels)
        loss_second.backward()
        optimizer.step()

        # Re-enable all gradients for both layer sets
        self._activate_layers(model, self.global_layer_names)
        self._activate_layers(model, self.local_layer_names)

        # Return the second loss for logging
        return loss_second


class LGFedAvgStepStrategyFromConfig(LGFedAvgStepStrategy):
    """
    LG-FedAvg strategy that reads layer names from Config.

    This variant automatically reads the global and local layer names from
    the configuration file, making it easier to use in existing Plato workflows.

    Configuration:
        The strategy looks for:
        - Config().algorithm.global_layer_names (required)
        - Config().algorithm.local_layer_names (required)
        - Config().algorithm.train_local_first (optional, default: True)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   global_layer_names:
        >>> #     - fc1
        >>> #     - fc2
        >>> #   local_layer_names:
        >>> #     - fc3
        >>> #   train_local_first: true
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyFromConfig
        >>>
        >>> trainer = ComposableTrainer(
        ...     training_step_strategy=LGFedAvgStepStrategyFromConfig()
        ... )
    """

    def __init__(self):
        """Initialize LG-FedAvg strategy from config."""
        config = Config()

        # Read layer names from config
        if not hasattr(config, "algorithm"):
            raise ValueError("Config must have 'algorithm' section for LG-FedAvg")

        if not hasattr(config.algorithm, "global_layer_names"):
            raise ValueError(
                "Config().algorithm.global_layer_names is required for LG-FedAvg"
            )

        if not hasattr(config.algorithm, "local_layer_names"):
            raise ValueError(
                "Config().algorithm.local_layer_names is required for LG-FedAvg"
            )

        global_layer_names = config.algorithm.global_layer_names
        local_layer_names = config.algorithm.local_layer_names

        # Optional: training order
        train_local_first = True
        if hasattr(config.algorithm, "train_local_first"):
            train_local_first = config.algorithm.train_local_first

        super().__init__(
            global_layer_names=global_layer_names,
            local_layer_names=local_layer_names,
            train_local_first=train_local_first,
        )


class LGFedAvgStepStrategyAuto(LGFedAvgStepStrategy):
    """
    LG-FedAvg strategy with automatic layer detection.

    This variant automatically determines which layers should be local vs global
    based on simple heuristics:
    - Last N layers are local (typically the classification head)
    - All other layers are global (typically the feature extractor)

    Args:
        num_local_layers: Number of layers to treat as local (from the end)
        train_local_first: If True, train local layers first

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyAuto
        >>>
        >>> # Automatically use last 1 layer as local
        >>> strategy = LGFedAvgStepStrategyAuto(num_local_layers=1)
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(self, num_local_layers: int = 1, train_local_first: bool = True):
        """
        Initialize auto LG-FedAvg strategy.

        Args:
            num_local_layers: Number of layers (from end) to treat as local
            train_local_first: If True, train local layers first
        """
        if num_local_layers < 1:
            raise ValueError("num_local_layers must be at least 1")

        self.num_local_layers = num_local_layers
        self._initialized = False

        # Initialize with dummy values, will be updated in setup()
        super().__init__(
            global_layer_names=["_temp_global"],
            local_layer_names=["_temp_local"],
            train_local_first=train_local_first,
        )

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the strategy by detecting layer names from the model.

        Args:
            context: Training context with model
        """
        # Get all parameter names
        param_names = [name for name, _ in context.model.named_parameters()]

        if len(param_names) == 0:
            raise ValueError("Model has no parameters")

        # Get unique layer names (remove '.weight', '.bias' suffixes)
        layer_names = []
        for name in param_names:
            # Extract layer name by removing weight/bias suffix
            if ".weight" in name:
                layer_name = name.replace(".weight", "")
            elif ".bias" in name:
                layer_name = name.replace(".bias", "")
            else:
                layer_name = name

            if layer_name not in layer_names:
                layer_names.append(layer_name)

        # Determine local vs global layers
        total_layers = len(layer_names)
        num_local = min(
            self.num_local_layers, total_layers - 1
        )  # Keep at least 1 global

        self.local_layer_names = layer_names[-num_local:]
        self.global_layer_names = layer_names[:-num_local]

        self._initialized = True

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """
        Perform training step, auto-detecting layers if not yet initialized.

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch
            labels: Target labels
            loss_criterion: Loss function
            context: Training context

        Returns:
            Loss value
        """
        if not self._initialized:
            self.setup(context)

        return super().training_step(
            model, optimizer, examples, labels, loss_criterion, context
        )
