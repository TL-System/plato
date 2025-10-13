"""
APFL Strategy Implementation

Reference:
Deng, Y., Kamani, M. M., & Mahdavi, M. (2020).
"Adaptive Personalized Federated Learning."
arXiv preprint arXiv:2003.13461.

Paper: https://arxiv.org/abs/2003.13461

Description:
APFL (Adaptive Personalized Federated Learning) maintains two models per client:
1. Global model (w): Received from server and updated via federated averaging
2. Personalized model (v): Kept locally and optimized for client's data

The key innovation is an adaptive mixing parameter α that determines the
interpolation between the two models:
    output = α * v + (1 - α) * w

The parameter α is learned adaptively for each client based on their local data,
allowing clients to determine the optimal balance between personalization and
global knowledge.

The update rules:
1. Update w using standard federated learning
2. Update v by optimizing: loss(α * v + (1 - α) * w) + regularization
3. Update α using gradient descent on the mixing objective
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import optimizers as optimizer_registry
from plato.trainers.strategies.base import (
    ModelUpdateStrategy,
    TrainingContext,
    TrainingStepStrategy,
)


class APFLUpdateStrategy(ModelUpdateStrategy):
    """
    APFL model update strategy for dual model management.

    This strategy manages the APFL-specific state:
    - Maintains a personalized model separate from the global model
    - Loads/saves personalized model and alpha parameter
    - Manages the adaptive mixing parameter α

    Args:
        alpha: Initial mixing parameter (default: 0.5).
               0 = fully personalized, 1 = fully global
        adaptive_alpha: If True, learns α adaptively (default: True)
        model_fn: Optional callable to create personalized model.
                  If None, uses models_registry.get()
        save_path: Optional custom path for saving models and alpha.
                   If None, uses Config().params["model_path"]

    Attributes:
        alpha: Current mixing parameter
        adaptive_alpha: Whether to adapt α
        personalized_model: The local personalized model
        personalized_optimizer: Optimizer for personalized model
        save_path: Path for saving state

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import (
        ...     APFLUpdateStrategy,
        ...     APFLStepStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=APFLUpdateStrategy(alpha=0.5),
        ...     training_step_strategy=APFLStepStrategy()
        ... )

    Note:
        This strategy should be used together with APFLStepStrategy which
        implements the dual model training logic.

        The learning rate for alpha updates is taken from the model's optimizer,
        matching the behavior of the original implementation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        adaptive_alpha: bool = True,
        model_fn: Optional[callable] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initialize APFL update strategy.

        Args:
            alpha: Initial mixing parameter (0 to 1)
            adaptive_alpha: Whether to learn α adaptively
            model_fn: Callable to create personalized model
            save_path: Custom path for saving state
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        self.model_fn = model_fn
        self.save_path = save_path
        self.personalized_model = None
        self.personalized_optimizer = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the strategy and create personalized model.

        Args:
            context: Training context with model and device
        """
        # Create personalized model
        if self.model_fn is None:
            self.personalized_model = models_registry.get()
        else:
            self.personalized_model = self.model_fn()

        # Determine save path
        if self.save_path is not None:
            base_path = self.save_path
        else:
            base_path = Config().params["model_path"]

        model_name = (
            Config().trainer.model_name
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "model_name")
            else "model"
        )

        self.personalized_model_path = (
            f"{base_path}/{model_name}_{context.client_id}_personalized_model.pth"
        )
        self.alpha_path = f"{base_path}/client_{context.client_id}_alpha.pth"

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Load personalized model and alpha at start of training.

        Args:
            context: Training context
        """
        # Load alpha if it exists
        if os.path.exists(self.alpha_path):
            try:
                self.alpha = torch.load(
                    self.alpha_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                logging.info(
                    "[Client #%d] Loaded APFL alpha: %.4f",
                    context.client_id,
                    self.alpha,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load alpha: %s", context.client_id, str(e)
                )

        # Load personalized model if it exists
        if os.path.exists(self.personalized_model_path):
            try:
                self.personalized_model.load_state_dict(
                    torch.load(
                        self.personalized_model_path,
                        map_location=torch.device("cpu"),
                    ),
                    strict=True,
                )
                logging.info(
                    "[Client #%d] Loaded APFL personalized model",
                    context.client_id,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load personalized model: %s",
                    context.client_id,
                    str(e),
                )

        # Move personalized model to device and set to training mode
        self.personalized_model.to(context.device)
        self.personalized_model.train()

        # Store in context for APFLStepStrategy
        context.state["apfl_personalized_model"] = self.personalized_model
        context.state["apfl_alpha"] = self.alpha
        context.state["apfl_adaptive_alpha"] = self.adaptive_alpha

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Save personalized model and alpha at end of training.

        Args:
            context: Training context
        """
        # Retrieve potentially updated alpha from context
        if "apfl_alpha" in context.state:
            self.alpha = context.state["apfl_alpha"]

        # Save alpha
        try:
            torch.save(self.alpha, self.alpha_path)
            logging.info(
                "[Client #%d] Saved APFL alpha: %.4f", context.client_id, self.alpha
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save alpha: %s", context.client_id, str(e)
            )

        # Save personalized model
        try:
            torch.save(
                self.personalized_model.state_dict(), self.personalized_model_path
            )
            logging.info(
                "[Client #%d] Saved APFL personalized model", context.client_id
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save personalized model: %s",
                context.client_id,
                str(e),
            )

        # Move to CPU to free GPU memory
        self.personalized_model.to(torch.device("cpu"))

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return empty payload (APFL only sends global model weights).

        Args:
            context: Training context

        Returns:
            Empty dictionary
        """
        return {}

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        self.personalized_model = None
        self.personalized_optimizer = None


class APFLStepStrategy(TrainingStepStrategy):
    """
    APFL training step strategy with dual model training.

    This strategy implements the APFL training procedure:
    1. Train global model with standard gradient descent
    2. Train personalized model by optimizing mixed output
    3. Optionally update mixing parameter α

    The training procedure for each step:
    1. Standard training step on global model (w)
    2. Forward pass through both models
    3. Compute mixed output: output = α * v + (1 - α) * w
    4. Compute loss on mixed output
    5. Backward pass to update personalized model (v)
    6. Update α if adaptive_alpha is True

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import (
        ...     APFLUpdateStrategy,
        ...     APFLStepStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=APFLUpdateStrategy(alpha=0.5),
        ...     training_step_strategy=APFLStepStrategy()
        ... )

    Note:
        This strategy requires APFLUpdateStrategy to set up the personalized
        model and alpha in the context.
    """

    def __init__(self):
        """Initialize APFL training step strategy."""
        pass

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
        Perform APFL training step with dual model training.

        Args:
            model: The global model (w)
            optimizer: Optimizer for the global model
            examples: Input batch
            labels: Target labels
            loss_criterion: Loss function
            context: Training context

        Returns:
            Loss value from personalized model training
        """
        # First: Standard training step on global model
        optimizer.zero_grad()
        outputs_global = model(examples)
        loss_global = loss_criterion(outputs_global, labels)
        loss_global.backward()
        optimizer.step()

        # Retrieve APFL-specific state from context
        personalized_model = context.state.get("apfl_personalized_model")
        alpha = context.state.get("apfl_alpha", 0.5)
        adaptive_alpha = context.state.get("apfl_adaptive_alpha", True)

        if personalized_model is None:
            # Fallback: just return global loss if no personalized model
            return loss_global

        # Get or create optimizer for personalized model
        if "apfl_personalized_optimizer" not in context.state:
            # Create optimizer for personalized model using the same configuration
            # as the global model optimizer (respects optimizer type and parameters)
            personalized_optimizer = optimizer_registry.get(personalized_model)
            context.state["apfl_personalized_optimizer"] = personalized_optimizer
        else:
            personalized_optimizer = context.state["apfl_personalized_optimizer"]

        # Second: Train personalized model with mixed output
        optimizer.zero_grad()
        personalized_optimizer.zero_grad()

        # Forward pass through both models
        outputs_personalized = personalized_model(examples)
        outputs_global_detached = model(examples)

        # Mixed output: α * v + (1 - α) * w
        mixed_output = (
            alpha * outputs_personalized + (1 - alpha) * outputs_global_detached
        )

        # Compute loss on mixed output
        personalized_loss = loss_criterion(mixed_output, labels)
        personalized_loss.backward()

        # Update personalized model
        personalized_optimizer.step()

        # Third: Update alpha if adaptive
        # Update alpha only once at the beginning of training (epoch 1, batch 0)
        # This matches the main branch behavior and the paper's algorithm
        current_batch = context.state.get("current_batch", 0)
        if adaptive_alpha and context.current_epoch == 1 and current_batch == 0:
            # Update alpha based on Eq. 10 in the paper
            # Get the current learning rate from the optimizer
            # (matches main branch behavior which uses lr_scheduler.get_lr()[0])
            current_lr = optimizer.param_groups[0]["lr"]
            alpha = self._update_alpha(
                model,
                personalized_model,
                alpha,
                current_lr,
                examples,
                labels,
                loss_criterion,
            )
            context.state["apfl_alpha"] = alpha

        return personalized_loss

    def _update_alpha(
        self,
        global_model: nn.Module,
        personalized_model: nn.Module,
        alpha: float,
        alpha_lr: float,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
    ) -> float:
        """
        Update mixing parameter α based on gradient.

        This implements Eq. 10 from the APFL paper:
        α_{t+1} = α_t - η_α * ∇_α L(α_t)

        Args:
            global_model: Global model
            personalized_model: Personalized model
            alpha: Current α value
            alpha_lr: Learning rate for α (typically the model's learning rate)
            examples: Input batch
            labels: Target labels
            loss_criterion: Loss function

        Returns:
            Updated α value (clipped to [0, 1])
        """
        # Compute gradient of α
        grad_alpha = 0.0

        for p_params, g_params in zip(
            personalized_model.parameters(), global_model.parameters()
        ):
            # Compute difference: v - w
            diff = p_params.data - g_params.data

            # Get gradient of mixed output
            if p_params.grad is not None and g_params.grad is not None:
                # Combined gradient: α * ∇v + (1 - α) * ∇w
                combined_grad = (
                    alpha * p_params.grad.data + (1 - alpha) * g_params.grad.data
                )

                # Gradient w.r.t. α: <diff, combined_grad>
                # Use torch.dot for 1D tensors to avoid deprecation warning
                grad_alpha += torch.dot(diff.view(-1), combined_grad.view(-1)).item()

        # Add L2 regularization on α (optional, coefficient 0.02 from paper)
        grad_alpha += 0.02 * alpha

        # Update α
        alpha_new = alpha - alpha_lr * grad_alpha

        # Clip to [0, 1]
        alpha_new = np.clip(alpha_new, 0.0, 1.0)

        return alpha_new


class APFLUpdateStrategyFromConfig(APFLUpdateStrategy):
    """
    APFL update strategy that reads configuration from Config.

    Configuration:
        - Config().algorithm.alpha (optional, default: 0.5)
        - Config().algorithm.adaptive_alpha (optional, default: True)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   alpha: 0.5
        >>> #   adaptive_alpha: true
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import (
        ...     APFLUpdateStrategyFromConfig,
        ...     APFLStepStrategy
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=APFLUpdateStrategyFromConfig(),
        ...     training_step_strategy=APFLStepStrategy()
        ... )
    """

    def __init__(self, model_fn: Optional[callable] = None):
        """
        Initialize APFL strategy from config.

        Args:
            model_fn: Optional callable to create personalized model
        """
        config = Config()

        # Read hyperparameters from config with defaults
        alpha = 0.5
        adaptive_alpha = True

        if hasattr(config, "algorithm"):
            if hasattr(config.algorithm, "alpha"):
                alpha = config.algorithm.alpha
            if hasattr(config.algorithm, "adaptive_alpha"):
                adaptive_alpha = config.algorithm.adaptive_alpha

        super().__init__(
            alpha=alpha,
            adaptive_alpha=adaptive_alpha,
            model_fn=model_fn,
        )
