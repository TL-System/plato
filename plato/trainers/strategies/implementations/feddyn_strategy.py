"""
FedDyn Strategy Implementation

Reference:
Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saligrama, V. (2021).
"Federated Learning Based on Dynamic Regularization."
In Proceedings of ICLR 2021.

Paper: https://openreview.net/forum?id=B7v4QMR6Z9w
Source code: https://github.com/alpemreacar/FedDyn

Description:
FedDyn addresses client drift by dynamically adjusting a regularization term that
accounts for the previous local model update. The local objective becomes:

    F_k(w) - <w, h_k> + (α/2)||w - w^t||^2

where:
- F_k(w) is the standard loss on client k's data
- h_k is a dynamic regularizer tracking previous local updates
- w^t is the global model at round t
- α is the regularization coefficient

The key difference from FedProx is that h_k changes based on local training history,
making the regularization adaptive.
"""

import copy
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    ModelUpdateStrategy,
    TrainingContext,
)


class FedDynLossStrategy(LossCriterionStrategy):
    """
    FedDyn loss strategy with dynamic regularization.

    This strategy implements the FedDyn local objective which includes:
    1. Standard task loss (e.g., cross-entropy)
    2. Linear penalty term: -<w, h_k>
    3. L2 regularization: (α/2)||w - w^t||^2

    The dynamic regularizer h_k is updated by the FedDynUpdateStrategy.

    Mathematical formulation:
        loss = task_loss - <w, h_k> + (α/2)||w - w^t||^2

    Args:
        alpha: Regularization coefficient (default: 0.01).
               Higher values enforce stronger proximity to global model.
        base_loss_fn: Base loss function. If None, uses CrossEntropyLoss.
        adaptive_alpha: If True, scales alpha by 1/weight where weight
                       is the relative data size of this client.

    Attributes:
        alpha: The regularization coefficient
        base_loss_fn: The underlying loss criterion
        adaptive_alpha: Whether to use adaptive alpha scaling
        global_model_weights: Snapshot of global model weights
        local_model_last_round: Previous round's local model weights

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedDynLossStrategy,
        ...     FedDynUpdateStrategy
        ... )
        >>>
        >>> # Create trainer with FedDyn
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedDynLossStrategy(alpha=0.01),
        ...     model_update_strategy=FedDynUpdateStrategy()
        ... )

    Note:
        FedDynLossStrategy should be used together with FedDynUpdateStrategy
        which manages the h_k dynamic regularizer state.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        base_loss_fn: Optional[callable] = None,
        adaptive_alpha: bool = True,
    ):
        """
        Initialize FedDyn loss strategy.

        Args:
            alpha: Regularization coefficient (typical: 0.001 to 0.1)
            base_loss_fn: Base loss function. If None, uses CrossEntropyLoss
            adaptive_alpha: Whether to scale alpha by client data weight
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.alpha = alpha
        self.base_loss_fn = base_loss_fn
        self.adaptive_alpha = adaptive_alpha
        self.global_model_weights = None
        self.local_model_last_round = None
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the loss strategy.

        Args:
            context: Training context
        """
        # Initialize base loss criterion
        if self.base_loss_fn is None:
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = self.base_loss_fn

        # Try to retrieve state from context
        self.global_model_weights = context.state.get("feddyn_global_weights")
        self.local_model_last_round = context.state.get("feddyn_local_last_round")

        # If not in context, initialize with current model
        if self.global_model_weights is None:
            self.global_model_weights = copy.deepcopy(context.model.state_dict())
            context.state["feddyn_global_weights"] = self.global_model_weights

        if self.local_model_last_round is None:
            self.local_model_last_round = copy.deepcopy(context.model.state_dict())
            context.state["feddyn_local_last_round"] = self.local_model_last_round

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute FedDyn loss: task loss + linear penalty + L2 regularization.

        The total loss is:
            loss = task_loss - <w, h_k> + (α/2)||w - w^t||^2

        where h_k = w_prev - w_global (maintained by FedDynUpdateStrategy)

        Args:
            outputs: Model predictions (logits)
            labels: Ground truth labels
            context: Training context with model access

        Returns:
            Scalar loss tensor combining all three terms
        """
        # Compute standard task loss
        task_loss = self._criterion(outputs, labels)

        # Get alpha coefficient (potentially adaptive)
        alpha_coef = self._get_alpha_coefficient(labels, context)

        # Compute linear penalty: -<w, h_k> where h_k = w_prev - w_global
        linear_penalty = torch.tensor(0.0, device=outputs.device)

        for name, param in context.model.named_parameters():
            if (
                name in self.local_model_last_round
                and name in self.global_model_weights
            ):
                w_prev = self.local_model_last_round[name].to(param.device)
                w_global = self.global_model_weights[name].to(param.device)
                h_k = w_prev - w_global

                # Compute dot product: <w, h_k>
                linear_penalty = linear_penalty + alpha_coef * torch.sum(param * h_k)

        # Compute L2 regularization: (α/2)||w - w^t||^2
        l2_reg = torch.tensor(0.0, device=outputs.device)

        for name, param in context.model.named_parameters():
            if name in self.global_model_weights:
                w_global = self.global_model_weights[name].to(param.device)
                l2_reg = l2_reg + torch.sum((param - w_global) ** 2)

        l2_reg = (alpha_coef / 2.0) * l2_reg

        # Total loss: task_loss - linear_penalty + l2_reg
        # Note: We subtract because in the formulation it's -<w, h_k>
        total_loss = task_loss - linear_penalty + l2_reg

        return total_loss

    def _get_alpha_coefficient(
        self, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Get alpha coefficient, potentially adapted by client data weight.

        Args:
            labels: Current batch labels
            context: Training context

        Returns:
            Alpha coefficient (scalar tensor)
        """
        if not self.adaptive_alpha:
            return torch.tensor(self.alpha, device=labels.device)

        # Compute weight list: proportion of data on this client
        # This is a simplified version - in practice, you'd need actual data sizes
        total_clients = (
            Config().clients.total_clients
            if hasattr(Config(), "clients")
            and hasattr(Config().clients, "total_clients")
            else 100
        )

        # Create uniform weight distribution
        weight_list = labels / torch.sum(labels) * total_clients

        # Adaptive alpha: α / weight (avoid division by zero)
        adaptive_alpha = self.alpha / torch.where(weight_list != 0, weight_list, 1.0)

        return torch.mean(adaptive_alpha).to(labels.device)

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        self.global_model_weights = None
        self.local_model_last_round = None


class FedDynUpdateStrategy(ModelUpdateStrategy):
    """
    FedDyn model update strategy for state management.

    This strategy manages the FedDyn-specific state:
    - Saves global model weights at start of training
    - Loads/saves local model from previous round
    - Provides state to FedDynLossStrategy

    Args:
        save_path: Optional custom path for saving local models.
                   If None, uses Config().params["model_path"]

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedDynLossStrategy,
        ...     FedDynUpdateStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedDynLossStrategy(alpha=0.01),
        ...     model_update_strategy=FedDynUpdateStrategy()
        ... )

    Note:
        This strategy should be used together with FedDynLossStrategy.
        The loss strategy accesses the state managed by this strategy.
    """

    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize FedDyn update strategy.

        Args:
            save_path: Optional custom path for saving local models
        """
        self.save_path = save_path
        self.global_model_weights = None
        self.local_model_last_round = None
        self.local_model_path = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the strategy and determine save path.

        Args:
            context: Training context with client_id
        """
        if self.save_path is not None:
            base_path = self.save_path
        else:
            base_path = Config().params["model_path"]

        model_name = (
            Config().trainer.model_name
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "model_name")
            else "model"
        )

        self.local_model_path = f"{base_path}_{model_name}_{context.client_id}.pth"

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Initialize FedDyn state at start of training round.

        This method:
        1. Saves current global model weights
        2. Loads previous round's local model if it exists
        3. Stores state in context for FedDynLossStrategy

        Args:
            context: Training context
        """
        # Save global model weights
        self.global_model_weights = copy.deepcopy(context.model.state_dict())

        # Try to load previous round's local model
        if os.path.exists(self.local_model_path):
            try:
                self.local_model_last_round = torch.load(
                    self.local_model_path, map_location=torch.device("cpu")
                )
                logging.info(
                    "[Client #%d] Loaded FedDyn local model from previous round: %s",
                    context.client_id,
                    self.local_model_path,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load previous local model: %s",
                    context.client_id,
                    str(e),
                )
                self.local_model_last_round = copy.deepcopy(self.global_model_weights)
        else:
            # First round: use global model as previous local model
            logging.info(
                "[Client #%d] No previous local model found. "
                "Using global model for first round.",
                context.client_id,
            )
            self.local_model_last_round = copy.deepcopy(self.global_model_weights)

        # Store in context for loss strategy
        context.state["feddyn_global_weights"] = self.global_model_weights
        context.state["feddyn_local_last_round"] = self.local_model_last_round

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Save local model at end of training round.

        Args:
            context: Training context
        """
        # Save current local model for next round
        try:
            torch.save(context.model.state_dict(), self.local_model_path)
            logging.info(
                "[Client #%d] Saved FedDyn local model to %s",
                context.client_id,
                self.local_model_path,
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save local model: %s",
                context.client_id,
                str(e),
            )

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return additional payload data (currently none for FedDyn).

        Args:
            context: Training context

        Returns:
            Empty dictionary (FedDyn only sends model weights)
        """
        return {}

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        self.global_model_weights = None
        self.local_model_last_round = None


class FedDynLossStrategyFromConfig(FedDynLossStrategy):
    """
    FedDyn loss strategy that reads configuration from Config.

    This variant automatically reads the alpha parameter from the configuration
    file, making it easier to use in existing Plato workflows.

    Configuration:
        The strategy looks for:
        - Config().algorithm.alpha_coef (preferred)
        - Config().algorithm.feddyn_alpha (fallback)
        - Default: 0.01 if neither is specified

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   alpha_coef: 0.01
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.implementations import (
        ...     FedDynLossStrategyFromConfig,
        ...     FedDynUpdateStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedDynLossStrategyFromConfig(),
        ...     model_update_strategy=FedDynUpdateStrategy()
        ... )
    """

    def __init__(
        self,
        base_loss_fn: Optional[callable] = None,
        adaptive_alpha: bool = True,
    ):
        """
        Initialize FedDyn loss strategy with config-based alpha.

        Args:
            base_loss_fn: Base loss function. If None, uses CrossEntropyLoss
            adaptive_alpha: Whether to scale alpha by client data weight
        """
        # Read alpha from config
        config = Config()
        alpha = 0.01  # default

        if hasattr(config, "algorithm") and hasattr(config.algorithm, "alpha_coef"):
            alpha = config.algorithm.alpha_coef
        elif hasattr(config, "algorithm") and hasattr(config.algorithm, "feddyn_alpha"):
            alpha = config.algorithm.feddyn_alpha

        super().__init__(
            alpha=alpha, base_loss_fn=base_loss_fn, adaptive_alpha=adaptive_alpha
        )
