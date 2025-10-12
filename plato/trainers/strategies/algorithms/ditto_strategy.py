"""
Ditto Strategy Implementation

Reference:
Li, T., Hu, S., Beirami, A., & Smith, V. (2021).
"Ditto: Fair and Robust Federated Learning Through Personalization."
In Proceedings of ICML 2021.

Paper: https://arxiv.org/abs/2012.04221

Description:
Ditto is a personalized federated learning algorithm that maintains two models:
1. Global model (w): Trained via standard federated learning
2. Personalized model (v): Trained locally with regularization towards global model

The key innovation is the personalized model training that happens after the
global model training. The personalized model optimizes:
    min_v F_k(v) + λ/2 * ||v - w||^2

where:
- F_k(v) is the local loss on client k's data
- w is the global model
- λ controls the strength of regularization towards the global model

Training procedure:
1. Train global model (w) with standard federated learning
2. After global training, train personalized model (v) with regularization
3. Send only global model to server; keep personalized model locally
4. Use personalized model for inference
"""

import copy
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import tracking
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext


class DittoUpdateStrategy(ModelUpdateStrategy):
    """
    Ditto model update strategy for personalized federated learning.

    This strategy manages the Ditto-specific state:
    - Maintains a personalized model separate from the global model
    - Trains personalized model after global model training completes
    - Applies regularization towards global model during personalization

    Args:
        ditto_lambda: Regularization coefficient for personalization (default: 0.1).
                      Higher values enforce stronger proximity to global model.
        personalization_epochs: Number of epochs for personalized training (default: 5)
        model_fn: Optional callable to create personalized model.
                  If None, uses models_registry.get()
        save_path: Optional custom path for saving personalized model.
                   If None, uses Config().params["model_path"]

    Attributes:
        ditto_lambda: Regularization coefficient
        personalization_epochs: Number of personalization epochs
        personalized_model: The local personalized model
        initial_global_weights: Global model weights at start of training
        save_path: Path for saving personalized model

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import DittoUpdateStrategy
        >>>
        >>> # Basic usage
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=DittoUpdateStrategy(ditto_lambda=0.1)
        ... )
        >>>
        >>> # With custom configuration
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=DittoUpdateStrategy(
        ...         ditto_lambda=0.1,
        ...         personalization_epochs=10
        ...     )
        ... )

    Note:
        The personalized model training happens in on_train_end(), after
        the global model training is complete. This means the trainer should
        have access to the training data during on_train_end().
    """

    def __init__(
        self,
        ditto_lambda: float = 0.1,
        personalization_epochs: int = 5,
        model_fn: Optional[callable] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initialize Ditto update strategy.

        Args:
            ditto_lambda: Regularization coefficient (typical: 0.01 to 1.0)
            personalization_epochs: Number of epochs for personalization
            model_fn: Callable to create personalized model
            save_path: Custom path for saving personalized model
        """
        if ditto_lambda < 0:
            raise ValueError(f"ditto_lambda must be non-negative, got {ditto_lambda}")

        if personalization_epochs < 1:
            raise ValueError(
                f"personalization_epochs must be at least 1, got {personalization_epochs}"
            )

        self.ditto_lambda = ditto_lambda
        self.personalization_epochs = personalization_epochs
        self.model_fn = model_fn
        self.save_path = save_path
        self.personalized_model = None
        self.initial_global_weights = None
        self.personalized_model_path = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the strategy and create personalized model.

        Args:
            context: Training context with model
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
            f"{base_path}/{model_name}_{context.client_id}_v_net.pth"
        )

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Save global model weights at start of training.

        Args:
            context: Training context with model
        """
        # Save initial global model weights (w^t in the paper)
        self.initial_global_weights = copy.deepcopy(context.model.cpu().state_dict())

        # Load existing personalized model if available
        if os.path.exists(self.personalized_model_path):
            try:
                self.personalized_model.load_state_dict(
                    torch.load(
                        self.personalized_model_path,
                        map_location=torch.device("cpu"),
                    )
                )
                logging.info(
                    "[Client #%d] Loaded existing Ditto personalized model",
                    context.client_id,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load personalized model: %s",
                    context.client_id,
                    str(e),
                )

        # Store in context
        context.state["ditto_initial_global_weights"] = self.initial_global_weights
        context.state["ditto_personalized_model"] = self.personalized_model

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Train personalized model after global model training completes.

        This implements Algorithm 1 from the Ditto paper, optimizing:
            v_{k+1} = v_k - η∇F_k(v_k) - ηλ(v_k - w^t)

        Args:
            context: Training context
        """
        logging.info(
            "[Client #%d] Starting Ditto personalized model training.",
            context.client_id,
        )

        # Get training data loader from context
        train_loader = context.state.get("train_loader")
        if train_loader is None:
            logging.warning(
                "[Client #%d] No train_loader found in context. "
                "Skipping personalized training.",
                context.client_id,
            )
            return

        # Create optimizer for personalized model
        lr = (
            Config().trainer.lr
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "lr")
            else 0.01
        )
        personalized_optimizer = torch.optim.SGD(
            self.personalized_model.parameters(), lr=lr
        )

        # Create learning rate scheduler
        personalized_scheduler = torch.optim.lr_scheduler.StepLR(
            personalized_optimizer, step_size=1, gamma=1.0
        )

        # Get loss criterion from context
        loss_criterion_fn = context.state.get("loss_criterion")
        if loss_criterion_fn is None:
            loss_criterion_fn = nn.CrossEntropyLoss()

        # Loss tracker
        epoch_loss_meter = tracking.LossTracker()

        # Move personalized model to device and set to training mode
        self.personalized_model.to(context.device)
        self.personalized_model.train()

        # Train personalized model for specified epochs
        for epoch in range(1, self.personalization_epochs + 1):
            epoch_loss_meter.reset()

            for batch_idx, (examples, labels) in enumerate(train_loader):
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                personalized_optimizer.zero_grad()

                # Forward pass
                outputs = self.personalized_model(examples)
                loss = loss_criterion_fn(outputs, labels)

                # Backward pass
                loss.backward()

                # Apply optimizer step
                personalized_optimizer.step()

                # Apply Ditto regularization: v -= ηλ(v - w^t)
                current_lr = personalized_optimizer.param_groups[0]["lr"]
                with torch.no_grad():
                    for v_name, v_param in self.personalized_model.named_parameters():
                        w_global = self.initial_global_weights[v_name].to(
                            context.device
                        )
                        # Regularization: v -= ηλ(v - w^t)
                        v_param.data = v_param.data - current_lr * self.ditto_lambda * (
                            v_param.data - w_global
                        )

                # Update loss tracker
                epoch_loss_meter.update(loss, labels.size(0))

            # Step the scheduler
            personalized_scheduler.step()

            logging.info(
                "[Client #%d] Ditto Personalization Epoch: [%d/%d]\tLoss: %.6f",
                context.client_id,
                epoch,
                self.personalization_epochs,
                epoch_loss_meter.average,
            )

        # Move personalized model back to CPU
        self.personalized_model.to(torch.device("cpu"))

        # Save personalized model
        try:
            torch.save(
                self.personalized_model.state_dict(), self.personalized_model_path
            )
            logging.info(
                "[Client #%d] Saved Ditto personalized model to %s",
                context.client_id,
                self.personalized_model_path,
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save personalized model: %s",
                context.client_id,
                str(e),
            )

        # Check if we're in final personalization round
        total_rounds = (
            Config().trainer.rounds
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "rounds")
            else 100
        )

        if context.current_round > total_rounds:
            # In final personalization round, use personalized model for inference
            logging.info(
                "[Client #%d] Using personalized model for final evaluation.",
                context.client_id,
            )
            # Copy personalized model weights to main model
            context.model.load_state_dict(self.personalized_model.state_dict())

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return empty payload (Ditto only sends global model weights).

        The personalized model is kept locally and never sent to server.

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
        self.initial_global_weights = None


class DittoUpdateStrategyFromConfig(DittoUpdateStrategy):
    """
    Ditto update strategy that reads configuration from Config.

    Configuration:
        - Config().algorithm.ditto_lambda (optional, default: 0.1)
        - Config().algorithm.personalization_epochs (optional, default: 5)
        - Config().trainer.epochs (used for personalization if not specified)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   ditto_lambda: 0.1
        >>> #   personalization_epochs: 10
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import DittoUpdateStrategyFromConfig
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=DittoUpdateStrategyFromConfig()
        ... )
    """

    def __init__(self, model_fn: Optional[callable] = None):
        """
        Initialize Ditto strategy from config.

        Args:
            model_fn: Optional callable to create personalized model
        """
        config = Config()

        # Read hyperparameters from config with defaults
        ditto_lambda = 0.1
        personalization_epochs = 5

        if hasattr(config, "algorithm"):
            if hasattr(config.algorithm, "ditto_lambda"):
                ditto_lambda = config.algorithm.ditto_lambda
            if hasattr(config.algorithm, "personalization_epochs"):
                personalization_epochs = config.algorithm.personalization_epochs
            elif hasattr(config, "trainer") and hasattr(config.trainer, "epochs"):
                # Fallback: use same number of epochs as training
                personalization_epochs = config.trainer.epochs

        super().__init__(
            ditto_lambda=ditto_lambda,
            personalization_epochs=personalization_epochs,
            model_fn=model_fn,
        )
