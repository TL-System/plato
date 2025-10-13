"""
Personalized Federated Learning Strategies Implementation

This module implements strategies for personalized federated learning algorithms
that maintain both global (shared) and local (personalized) model components.

References:
1. FedPer: "Federated Learning with Personalization Layers"
   Arivazhagan et al., 2019.
   Paper: https://arxiv.org/abs/1912.00818

2. FedRep: "Exploiting Shared Representations for Personalized Federated Learning"
   Collins et al., ICML 2021.
   Paper: https://arxiv.org/abs/2102.07078

Description:
Both FedPer and FedRep divide the model into:
- Global/shared layers: Trained collaboratively and aggregated on server
- Local/personalized layers: Trained locally and never shared

The key difference:
- FedPer: Simply freezes global layers during final personalization rounds
- FedRep: Alternates training between local and global layers during regular rounds
"""

from typing import List, Optional

from plato.config import Config
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext


class FedPerUpdateStrategy(ModelUpdateStrategy):
    """
    FedPer personalization strategy.

    FedPer (Federated Learning with Personalization Layers) maintains global
    and local layers. During regular federated learning rounds, all layers are
    trained. During final personalization rounds, global layers are frozen and
    only local layers are trained.

    Args:
        global_layer_names: List of layer name patterns for global layers
        personalization_rounds: Number of rounds for final personalization.
                               If None, reads from Config().trainer.rounds

    Attributes:
        global_layer_names: Patterns for identifying global layers
        personalization_rounds: Number of final personalization rounds
        is_personalizing: Flag indicating if in personalization phase

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedPerUpdateStrategy
        >>>
        >>> # Specify global layers explicitly
        >>> strategy = FedPerUpdateStrategy(
        ...     global_layer_names=['conv', 'fc1', 'fc2']
        ... )
        >>> trainer = ComposableTrainer(model_update_strategy=strategy)
        >>>
        >>> # Use config-based variant
        >>> strategy = FedPerUpdateStrategyFromConfig()

    Note:
        The trainer should set context.current_round to track training progress.
        Personalization begins when current_round > trainer.rounds.
    """

    def __init__(
        self,
        global_layer_names: List[str],
        personalization_rounds: Optional[int] = None,
    ):
        """
        Initialize FedPer update strategy.

        Args:
            global_layer_names: List of layer name patterns for global layers
            personalization_rounds: Number of final personalization rounds
        """
        if not global_layer_names:
            raise ValueError("global_layer_names cannot be empty")

        self.global_layer_names = global_layer_names
        self.personalization_rounds = personalization_rounds
        self.is_personalizing = False

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Determine if in personalization phase and freeze global layers if so.

        Args:
            context: Training context with current_round
        """
        # Determine total rounds
        total_rounds = (
            Config().trainer.rounds
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "rounds")
            else 100
        )

        # Check if we're in personalization phase
        self.is_personalizing = context.current_round > total_rounds

        if self.is_personalizing:
            # Freeze global layers during personalization
            self._freeze_global_layers(context)

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Re-activate global layers after training.

        Args:
            context: Training context
        """
        if self.is_personalizing:
            # Re-activate global layers
            self._activate_global_layers(context)

    def _freeze_global_layers(self, context: TrainingContext) -> None:
        """
        Freeze global layers (disable gradients).

        Args:
            context: Training context with model
        """
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.global_layer_names):
                param.requires_grad = False

    def _activate_global_layers(self, context: TrainingContext) -> None:
        """
        Activate global layers (enable gradients).

        Args:
            context: Training context with model
        """
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.global_layer_names):
                param.requires_grad = True


class FedPerUpdateStrategyFromConfig(FedPerUpdateStrategy):
    """
    FedPer strategy that reads configuration from Config.

    Configuration:
        - Config().algorithm.global_layer_names (required)
        - Config().trainer.rounds (for determining personalization phase)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   global_layer_names:
        >>> #     - conv
        >>> #     - fc1
        >>> # trainer:
        >>> #   rounds: 100
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedPerUpdateStrategyFromConfig
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=FedPerUpdateStrategyFromConfig()
        ... )
    """

    def __init__(self):
        """Initialize FedPer strategy from config."""
        config = Config()

        if not hasattr(config, "algorithm"):
            raise ValueError("Config must have 'algorithm' section for FedPer")

        if not hasattr(config.algorithm, "global_layer_names"):
            raise ValueError(
                "Config().algorithm.global_layer_names is required for FedPer"
            )

        global_layer_names = config.algorithm.global_layer_names

        super().__init__(global_layer_names=global_layer_names)


class FedRepUpdateStrategy(ModelUpdateStrategy):
    """
    FedRep personalization strategy.

    FedRep (Federated Learning with Representation Learning) alternates between
    training local and global layers during regular federated learning rounds.
    This allows better separation of personalized and shared representations.

    Training procedure:
    - Regular rounds (round <= total_rounds):
        - Train local layers for local_epochs
        - Train global layers for remaining epochs
    - Personalization rounds (round > total_rounds):
        - Freeze global layers, train only local layers

    Args:
        global_layer_names: List of layer name patterns for global layers
        local_layer_names: List of layer name patterns for local layers
        local_epochs: Number of epochs to train local layers first

    Attributes:
        global_layer_names: Patterns for identifying global layers
        local_layer_names: Patterns for identifying local layers
        local_epochs: Number of epochs for local layer training
        is_personalizing: Flag indicating if in personalization phase
        original_epochs: Original number of training epochs

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedRepUpdateStrategy
        >>>
        >>> strategy = FedRepUpdateStrategy(
        ...     global_layer_names=['conv', 'fc1'],
        ...     local_layer_names=['fc2'],
        ...     local_epochs=2
        ... )
        >>> trainer = ComposableTrainer(model_update_strategy=strategy)

    Note:
        FedRep requires epoch-level control. If your trainer doesn't track
        current_epoch in context, this strategy may not work as expected.
    """

    def __init__(
        self,
        global_layer_names: List[str],
        local_layer_names: List[str],
        local_epochs: int = 1,
    ):
        """
        Initialize FedRep update strategy.

        Args:
            global_layer_names: List of layer name patterns for global layers
            local_layer_names: List of layer name patterns for local layers
            local_epochs: Number of epochs to train local layers first
        """
        if not global_layer_names:
            raise ValueError("global_layer_names cannot be empty")
        if not local_layer_names:
            raise ValueError("local_layer_names cannot be empty")
        if local_epochs < 1:
            raise ValueError("local_epochs must be at least 1")

        self.global_layer_names = global_layer_names
        self.local_layer_names = local_layer_names
        self.local_epochs = local_epochs
        self.is_personalizing = False
        self.original_epochs = None
        self._last_processed_epoch = None

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Determine training phase and adjust configuration.

        Args:
            context: Training context
        """
        # Reset epoch tracking
        self._last_processed_epoch = None

        # Determine total rounds
        total_rounds = (
            Config().trainer.rounds
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "rounds")
            else 100
        )

        # Check if we're in personalization phase
        self.is_personalizing = context.current_round > total_rounds

        if self.is_personalizing:
            # During personalization, freeze global layers
            self._freeze_global_layers(context)

            # Optionally adjust epochs for personalization
            if hasattr(Config(), "algorithm") and hasattr(
                Config().algorithm, "personalization"
            ):
                if hasattr(Config().algorithm.personalization, "epochs"):
                    # Modify the training config to use personalization epochs
                    context.config["epochs"] = Config().algorithm.personalization.epochs

    def before_step(self, context: TrainingContext) -> None:
        """
        Adjust layer freezing based on current epoch during regular training.

        During regular rounds:
        - Epochs 1 to local_epochs: Train local layers only
        - Remaining epochs: Train global layers only

        Args:
            context: Training context with current_epoch
        """
        if not self.is_personalizing:
            current_epoch = context.current_epoch

            # Optimize: only update layer freezing when epoch changes
            if self._last_processed_epoch != current_epoch:
                self._last_processed_epoch = current_epoch

                if current_epoch <= self.local_epochs:
                    # Train local layers, freeze global layers
                    self._freeze_global_layers(context)
                    self._activate_local_layers(context)
                else:
                    # Train global layers, freeze local layers
                    self._freeze_local_layers(context)
                    self._activate_global_layers(context)

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Re-activate all layers after training.

        Args:
            context: Training context
        """
        # Re-activate all layers
        self._activate_global_layers(context)
        self._activate_local_layers(context)

    def _freeze_global_layers(self, context: TrainingContext) -> None:
        """Freeze global layers."""
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.global_layer_names):
                param.requires_grad = False

    def _activate_global_layers(self, context: TrainingContext) -> None:
        """Activate global layers."""
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.global_layer_names):
                param.requires_grad = True

    def _freeze_local_layers(self, context: TrainingContext) -> None:
        """Freeze local layers."""
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.local_layer_names):
                param.requires_grad = False

    def _activate_local_layers(self, context: TrainingContext) -> None:
        """Activate local layers."""
        for name, param in context.model.named_parameters():
            if any(layer_name in name for layer_name in self.local_layer_names):
                param.requires_grad = True


class FedRepUpdateStrategyFromConfig(FedRepUpdateStrategy):
    """
    FedRep strategy that reads configuration from Config.

    Configuration:
        - Config().algorithm.global_layer_names (required)
        - Config().algorithm.local_layer_names (required)
        - Config().algorithm.local_epochs (optional, default: 1)

    Example:
        >>> # In config file:
        >>> # algorithm:
        >>> #   global_layer_names:
        >>> #     - conv
        >>> #     - fc1
        >>> #   local_layer_names:
        >>> #     - fc2
        >>> #   local_epochs: 2
        >>>
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import FedRepUpdateStrategyFromConfig
        >>>
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=FedRepUpdateStrategyFromConfig()
        ... )
    """

    def __init__(self):
        """Initialize FedRep strategy from config."""
        config = Config()

        if not hasattr(config, "algorithm"):
            raise ValueError("Config must have 'algorithm' section for FedRep")

        if not hasattr(config.algorithm, "global_layer_names"):
            raise ValueError(
                "Config().algorithm.global_layer_names is required for FedRep"
            )

        if not hasattr(config.algorithm, "local_layer_names"):
            raise ValueError(
                "Config().algorithm.local_layer_names is required for FedRep"
            )

        global_layer_names = config.algorithm.global_layer_names
        local_layer_names = config.algorithm.local_layer_names

        # Optional local_epochs
        local_epochs = 1
        if hasattr(config.algorithm, "local_epochs"):
            local_epochs = config.algorithm.local_epochs

        super().__init__(
            global_layer_names=global_layer_names,
            local_layer_names=local_layer_names,
            local_epochs=local_epochs,
        )
