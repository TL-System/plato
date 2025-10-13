"""
Optimizer strategy for Calibre that handles both SSL training and personalization.

During the SSL training phase (first rounds), the optimizer uses standard settings.
During the personalization phase (after Config().trainer.rounds), the optimizer
adds the encoder parameters as an additional parameter group, allowing the entire
model (encoder + linear classifier) to be used during inference.
"""

import torch.nn as nn
import torch.optim as optim

from plato.config import Config
from plato.trainers import optimizers
from plato.trainers.strategies.base import OptimizerStrategy, TrainingContext


class CalibreOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy for Calibre that handles SSL training and personalization phases.

    During SSL training (rounds 1 to Config().trainer.rounds):
    - Creates optimizer from config for the main model

    During personalization (after Config().trainer.rounds):
    - Creates optimizer for local layers (linear classifier)
    - Adds encoder parameters as a separate parameter group
    - Encoder is not trained but included for the complete model structure

    This allows the trained SSL encoder to be combined with the locally trained
    classifier during the personalization phase.
    """

    def __init__(self):
        """Initialize the Calibre optimizer strategy."""
        self._local_layers = None

    def setup(self, context: TrainingContext):
        """
        Initialize local layers for personalization.

        Args:
            context: Training context with model and config
        """
        # Store local layers if they exist in the model
        if hasattr(context.model, "local_layers"):
            self._local_layers = context.model.local_layers

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> optim.Optimizer:
        """
        Create optimizer based on the current training phase.

        Args:
            model: The model to optimize
            context: Training context with current round information

        Returns:
            Configured optimizer for the current phase
        """
        current_round = context.current_round

        # Check if we're in the personalization phase
        if current_round > Config().trainer.rounds:
            return self._create_personalization_optimizer(model, context)
        else:
            return self._create_ssl_optimizer(model, context)

    def _create_ssl_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> optim.Optimizer:
        """
        Create optimizer for SSL training phase.

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            Standard optimizer from config
        """
        # Use the default optimizer from config
        optimizer_name = (
            Config().trainer.optimizer
            if hasattr(Config().trainer, "optimizer")
            else "SGD"
        )

        if hasattr(Config().parameters, "optimizer"):
            optimizer_params = Config().parameters.optimizer._asdict()
        else:
            optimizer_params = {"lr": 0.01}

        return optimizers.get(
            model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def _create_personalization_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> optim.Optimizer:
        """
        Create optimizer for personalization phase.

        During personalization:
        1. Creates optimizer for local layers (the classifier)
        2. Adds encoder parameters as an additional parameter group
        3. This allows the complete model (encoder + classifier) to be used

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            Optimizer configured for personalization
        """
        # Get personalization optimizer settings from config
        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()

        # Get local layers (linear classifier) from context or model
        local_layers = context.state.get("local_layers")
        if local_layers is None:
            # Try to get from model
            if hasattr(model, "local_layers"):
                local_layers = model.local_layers
            elif self._local_layers is not None:
                local_layers = self._local_layers
            else:
                # Fallback to optimizing the entire model
                return optimizers.get(
                    model,
                    optimizer_name=optimizer_name,
                    optimizer_params=optimizer_params,
                )

        # Create optimizer for local layers
        optimizer = optimizers.get(
            local_layers,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

        # Add encoder parameters as an additional parameter group
        # Note: These are not trained but included to build the complete model structure
        if hasattr(model, "encoder"):
            optimizer.add_param_group({"params": model.encoder.parameters()})

        return optimizer
