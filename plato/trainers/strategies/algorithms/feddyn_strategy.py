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
accounts for cumulative local model updates. The local objective becomes:

    min_θ [L_k(θ) - <∇L_k(θ_k^{t-1}), θ> + (α/2)||θ - θ^{t-1}||^2]

where:
- L_k(θ) is the local loss on client k's data
- ∇L_k(θ_k^{t-1}) is a cumulative dynamic regularizer (gradient vector)
- θ^{t-1} is the global model at round t-1
- α is the regularization coefficient

The dynamic regularizer is updated after training:
    ∇L_k(θ_k^t) = ∇L_k(θ_k^{t-1}) - α(θ_k^t - θ^{t-1})

This cumulative tracking of historical updates is the key innovation that makes
FedDyn different from FedProx and other methods.
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
    FedDyn loss strategy with cumulative dynamic regularization.

    This strategy implements the FedDyn local objective which includes:
    1. Standard task loss (e.g., cross-entropy)
    2. Linear penalty term: <w, -w_global + grad_vector>
    3. L2 regularization: (α/2)||w - w_global||^2

    The cumulative gradient vector grad_vector is maintained across rounds:
        grad_vector += (w_trained - w_global) after each training round

    Mathematical formulation (from paper):
        loss = task_loss + α * <w, -w_global + grad_vector> + (α/2)||w - w_global||^2

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
        cumulative_grad_vector: Cumulative gradient vector tracking historical updates

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import (
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
        which manages the cumulative gradient vector state across rounds.
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
        self.cumulative_grad_vector = None
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
        self.cumulative_grad_vector = context.state.get("feddyn_cumulative_grad")

        # If not in context, initialize
        if self.global_model_weights is None:
            self.global_model_weights = copy.deepcopy(context.model.state_dict())
            context.state["feddyn_global_weights"] = self.global_model_weights

        if self.cumulative_grad_vector is None:
            # Initialize cumulative gradient vector to zero
            self.cumulative_grad_vector = {
                name: torch.zeros_like(param)
                for name, param in context.model.named_parameters()
            }
            context.state["feddyn_cumulative_grad"] = self.cumulative_grad_vector

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute FedDyn loss with cumulative dynamic regularization.

        The total loss is (following the original paper and GitHub implementation):
            loss = task_loss + α * <w, -w_global + grad_vector> + (α/2)||w - w_global||^2

        where grad_vector is the cumulative sum of (w_trained - w_global) across rounds.

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

        # Compute linear penalty: α * <w, -w_global + grad_vector>
        linear_penalty = torch.tensor(0.0, device=outputs.device)

        for name, param in context.model.named_parameters():
            if (
                name in self.cumulative_grad_vector
                and name in self.global_model_weights
            ):
                grad_vec = self.cumulative_grad_vector[name].to(param.device)
                w_global = self.global_model_weights[name].to(param.device)

                # Compute: <w, -w_global + grad_vector>
                linear_penalty = linear_penalty + torch.sum(
                    param * (-w_global + grad_vec)
                )

        linear_penalty = alpha_coef * linear_penalty

        # Compute L2 regularization: (α/2)||w - w_global||^2
        l2_reg = torch.tensor(0.0, device=outputs.device)

        for name, param in context.model.named_parameters():
            if name in self.global_model_weights:
                w_global = self.global_model_weights[name].to(param.device)
                l2_reg = l2_reg + torch.sum((param - w_global) ** 2)

        l2_reg = (alpha_coef / 2.0) * l2_reg

        # Total loss: task_loss + linear_penalty + l2_reg
        # Note: We add linear_penalty because it already includes the sign
        total_loss = task_loss + linear_penalty + l2_reg

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
        self.cumulative_grad_vector = None


class FedDynUpdateStrategy(ModelUpdateStrategy):
    """
    FedDyn model update strategy for cumulative gradient state management.

    This strategy manages the FedDyn-specific cumulative state:
    - Saves global model weights at start of training
    - Loads/saves cumulative gradient vector across rounds
    - Updates gradient vector after training: grad_vec += (w_trained - w_global)
    - Provides state to FedDynLossStrategy

    The cumulative gradient vector is the key to FedDyn's dynamic regularization,
    tracking the sum of all historical local model deviations from global models.

    Args:
        save_path: Optional custom path for saving gradient vectors.
                   If None, uses Config().params["model_path"]

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import (
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
        The loss strategy accesses the cumulative gradient vector managed by this strategy.
    """

    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize FedDyn update strategy.

        Args:
            save_path: Optional custom path for saving gradient vectors
        """
        self.save_path = save_path
        self.global_model_weights = None
        self.cumulative_grad_vector = None
        self.grad_vector_path = None

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

        # Path for saving cumulative gradient vector
        self.grad_vector_path = f"{base_path}_feddyn_grad_{context.client_id}.pth"

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Initialize FedDyn state at start of training round.

        This method:
        1. Saves current global model weights
        2. Loads cumulative gradient vector from previous rounds if it exists
        3. Stores state in context for FedDynLossStrategy

        Args:
            context: Training context
        """
        # Save global model weights at start of this round
        self.global_model_weights = copy.deepcopy(context.model.state_dict())

        # Try to load cumulative gradient vector from previous rounds
        if os.path.exists(self.grad_vector_path):
            try:
                self.cumulative_grad_vector = torch.load(
                    self.grad_vector_path, map_location=torch.device("cpu")
                )
                logging.info(
                    "[Client #%d] Loaded FedDyn cumulative gradient vector from: %s",
                    context.client_id,
                    self.grad_vector_path,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load cumulative gradient vector: %s",
                    context.client_id,
                    str(e),
                )
                # Initialize to zero if loading fails
                self.cumulative_grad_vector = {
                    name: torch.zeros_like(param)
                    for name, param in context.model.named_parameters()
                }
        else:
            # First round: initialize cumulative gradient vector to zero
            logging.info(
                "[Client #%d] No previous gradient vector found. "
                "Initializing to zero for first round.",
                context.client_id,
            )
            self.cumulative_grad_vector = {
                name: torch.zeros_like(param)
                for name, param in context.model.named_parameters()
            }

        # Store in context for loss strategy
        context.state["feddyn_global_weights"] = self.global_model_weights
        context.state["feddyn_cumulative_grad"] = self.cumulative_grad_vector

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Update and save cumulative gradient vector at end of training round.

        This implements the key FedDyn update:
            grad_vector += (w_trained - w_global)

        Args:
            context: Training context
        """
        # Update cumulative gradient vector: grad_vec += (w_trained - w_global)
        trained_weights = context.model.state_dict()

        for name in self.cumulative_grad_vector:
            if name in trained_weights and name in self.global_model_weights:
                # Compute the difference: w_trained - w_global (both on CPU)
                trained_param_cpu = trained_weights[name].cpu()
                global_param_cpu = self.global_model_weights[name].cpu()
                diff = trained_param_cpu - global_param_cpu
                # Add to cumulative gradient vector
                self.cumulative_grad_vector[name] = (
                    self.cumulative_grad_vector[name] + diff
                )

        # Save updated cumulative gradient vector for next round
        try:
            torch.save(self.cumulative_grad_vector, self.grad_vector_path)
            logging.info(
                "[Client #%d] Updated and saved FedDyn cumulative gradient vector to %s.",
                context.client_id,
                self.grad_vector_path,
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save cumulative gradient vector: %s",
                context.client_id,
                str(e),
            )

        # Update state in context for next potential use
        context.state["feddyn_cumulative_grad"] = self.cumulative_grad_vector

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
        self.cumulative_grad_vector = None


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
        >>> from plato.trainers.strategies.algorithms import (
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
