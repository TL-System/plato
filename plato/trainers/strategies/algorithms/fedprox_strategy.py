"""
FedProx Strategy Implementation

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks."
Proceedings of Machine Learning and Systems (MLSys), vol. 2, 429-450.

Paper: https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf

Description:
FedProx addresses system heterogeneity in federated learning by adding a proximal
term to the local objective function. This helps prevent client drift and improves
convergence in heterogeneous settings.

The local objective becomes:
    h_k(w; w^t) = F_k(w) + (mu/2)||w - w^t||

where:
- F_k(w) is the standard loss function on client k's data
- w^t is the global model at round t
- mu is the proximal term coefficient

Note: This implementation uses the L2 norm (not squared) for backward compatibility
with the original Plato implementation, although the paper formula shows ||w - w^t||^2.
"""

from typing import Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


class FedProxLossStrategy(LossCriterionStrategy):
    """
    FedProx loss strategy with proximal term regularization.

    This strategy adds a proximal term to the base loss function to prevent
    client models from diverging too far from the global model. This is
    particularly useful in heterogeneous federated learning settings.

    Mathematical formulation:
        loss = base_loss(outputs, labels) + (mu/2) * ||w - w_global||

    Note: This implementation uses the L2 norm (not squared) for backward
    compatibility with the original Plato FedProx implementation.

    Args:
        mu: Proximal term penalty coefficient (default: 0.01).
            Higher values enforce stronger proximity to global model.
        base_loss_fn: Base loss function to use. If None, uses CrossEntropyLoss.
        norm_type: Type of norm to use for proximal term ('l2' or 'l1').

    Attributes:
        mu: The proximal term coefficient
        base_loss_fn: The underlying loss criterion
        global_weights: Dictionary storing global model weights
        norm_type: The norm type for computing the proximal term

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedProxLossStrategy
        >>>
        >>> # Create trainer with FedProx loss
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedProxLossStrategy(mu=0.01)
        ... )
        >>>
        >>> # With custom base loss
        >>> import torch.nn as nn
        >>> custom_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedProxLossStrategy(mu=0.1, base_loss_fn=custom_loss)
        ... )

    Note:
        The global model weights are captured during setup() and remain fixed
        throughout the local training round. They should be updated at the
        start of each federated learning round.
    """

    def __init__(
        self,
        mu: float = 0.01,
        base_loss_fn: Optional[callable] = None,
        norm_type: str = "l2",
    ):
        """
        Initialize FedProx loss strategy.

        Args:
            mu: Proximal term penalty coefficient. Typical values: 0.001 to 0.1
            base_loss_fn: Base loss function. If None, uses CrossEntropyLoss
            norm_type: Norm type for proximal term ('l2' or 'l1')
        """
        if mu < 0:
            raise ValueError(f"mu must be non-negative, got {mu}")

        if norm_type not in ["l2", "l1"]:
            raise ValueError(f"norm_type must be 'l2' or 'l1', got {norm_type}")

        self.mu = mu
        self.base_loss_fn = base_loss_fn
        self.norm_type = norm_type
        self.global_weights = None
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the loss strategy and capture global model weights.

        This method is called once at the start of training. It:
        1. Initializes the base loss criterion
        2. Captures a snapshot of the global model weights

        Args:
            context: Training context containing model and device info

        Note:
            The global weights are cloned and detached to prevent
            gradient computation through them.
        """
        # Initialize base loss criterion
        if self.base_loss_fn is None:
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = self.base_loss_fn

        # Capture global model weights at start of training
        # These represent w^t in the FedProx formulation
        self.global_weights = {}
        for name, param in context.model.named_parameters():
            self.global_weights[name] = param.clone().detach()

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute FedProx loss: base loss + proximal term.

        The total loss is:
            loss = base_loss + (mu/2) * ||w - w_global||

        Args:
            outputs: Model predictions (logits)
            labels: Ground truth labels
            context: Training context with model access

        Returns:
            Scalar loss tensor combining base loss and proximal term

        Note:
            The proximal term is computed only for parameters that have
            gradients enabled and exist in the global_weights dictionary.
            This implementation uses the L2 norm (not squared) to match
            the behavior of the original Plato implementation.
        """
        # Compute base loss (e.g., cross-entropy)
        base_loss = self._criterion(outputs, labels)

        # Compute proximal term: (mu/2) * ||w - w_global||
        # Note: We use L2 norm (not squared) for backward compatibility
        squared_diff_sum = torch.tensor(0.0, device=outputs.device)

        for name, param in context.model.named_parameters():
            if param.requires_grad and name in self.global_weights:
                global_param = self.global_weights[name].to(param.device)

                if self.norm_type == "l2":
                    # Sum of squared differences for L2 norm computation
                    squared_diff_sum = squared_diff_sum + torch.sum(
                        (param - global_param) ** 2
                    )
                else:  # l1
                    # L1 norm: ||w - w_global||
                    squared_diff_sum = squared_diff_sum + torch.sum(
                        torch.abs(param - global_param)
                    )

        # Compute the actual norm and scale by mu/2
        if self.norm_type == "l2":
            # Take square root to get L2 norm (not squared L2 norm)
            proximal_term = (self.mu / 2.0) * torch.sqrt(squared_diff_sum)
        else:
            proximal_term = self.mu * squared_diff_sum

        # Total loss
        total_loss = base_loss + proximal_term

        return total_loss

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        # Clear global weights to free memory
        if self.global_weights is not None:
            self.global_weights.clear()
            self.global_weights = None


class FedProxLossStrategyFromConfig(FedProxLossStrategy):
    """
    FedProx loss strategy that reads configuration from Config.

    This variant automatically reads the mu parameter from the configuration
    file, making it easier to use in existing Plato workflows.

    Configuration:
        The strategy looks for:
        - Config().clients.proximal_term_penalty_constant
        - Config().algorithm.fedprox_mu (fallback)
        - Default: 0.01 if neither is specified

    Example:
        >>> # In config file:
        >>> # clients:
        >>> #   proximal_term_penalty_constant: 0.1
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedProxLossStrategyFromConfig
        >>>
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedProxLossStrategyFromConfig()
        ... )
    """

    def __init__(self, base_loss_fn: Optional[callable] = None, norm_type: str = "l2"):
        """
        Initialize FedProx loss strategy with config-based mu.

        Args:
            base_loss_fn: Base loss function. If None, uses CrossEntropyLoss
            norm_type: Norm type for proximal term ('l2' or 'l1')
        """
        # Read mu from config
        config = Config()
        mu = 0.01  # default

        if hasattr(config, "clients") and hasattr(
            config.clients, "proximal_term_penalty_constant"
        ):
            mu = config.clients.proximal_term_penalty_constant
        elif hasattr(config, "algorithm") and hasattr(config.algorithm, "fedprox_mu"):
            mu = config.algorithm.fedprox_mu

        super().__init__(mu=mu, base_loss_fn=base_loss_fn, norm_type=norm_type)
