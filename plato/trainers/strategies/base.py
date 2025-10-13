"""
Base strategy interfaces for composable trainer architecture.

This module defines the core strategy interfaces and TrainingContext for
the composition-based trainer design. Instead of using inheritance to extend
trainer functionality, strategies are injected as dependencies.

Example:
    >>> from plato.trainers.composable import ComposableTrainer
    >>> from plato.trainers.strategies.algorithms import FedProxLossStrategy
    >>>
    >>> trainer = ComposableTrainer(
    ...     loss_strategy=FedProxLossStrategy(mu=0.01)
    ... )
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TrainingContext:
    """
    Shared context passed between strategies during training.

    The TrainingContext acts as a data container that allows strategies to:
    - Access common training state (model, device, etc.)
    - Share data between strategies via the `state` dictionary
    - Communicate information across training lifecycle

    Attributes:
        model: The neural network model being trained
        device: The torch device (CPU or GPU)
        client_id: ID of the client (0 for server)
        current_epoch: Current training epoch number (1-indexed)
        current_round: Current federated learning round number (1-indexed)
        config: Training configuration dictionary
        state: Dictionary for strategies to share arbitrary state data

    Example:
        >>> context = TrainingContext()
        >>> context.model = nn.Linear(10, 2)
        >>> context.device = torch.device('cuda')
        >>> context.state['custom_data'] = some_value
    """

    def __init__(self):
        """Initialize training context with default values."""
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.client_id: int = 0
        self.current_epoch: int = 0
        self.current_round: int = 0
        self.config: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

    def __repr__(self) -> str:
        """Return string representation of context."""
        return (
            f"TrainingContext(client_id={self.client_id}, "
            f"epoch={self.current_epoch}, round={self.current_round})"
        )


class Strategy(ABC):
    """
    Base class for all training strategies.

    All strategies inherit from this base class and can implement
    setup/teardown lifecycle methods for initialization and cleanup.

    The strategy pattern allows algorithms to be swapped at runtime
    without changing the trainer implementation.
    """

    def setup(self, context: TrainingContext) -> None:
        """
        Called once during trainer initialization.

        Use this method to:
        - Initialize strategy state
        - Access model architecture
        - Allocate resources
        - Load saved state from disk

        Args:
            context: Training context with model, device, etc.
        """
        pass

    def teardown(self, context: TrainingContext) -> None:
        """
        Called when all training is complete.

        Use this method to:
        - Clean up resources
        - Save final state to disk
        - Release memory

        Args:
            context: Training context
        """
        pass


class LossCriterionStrategy(Strategy):
    """
    Strategy interface for computing loss during training.

    Implement this interface to customize how loss is computed.
    Common use cases include:
    - Adding regularization terms (FedProx proximal term)
    - Custom loss functions
    - Multi-task losses
    - Adaptive weighting

    Example:
        >>> class MyLossStrategy(LossCriterionStrategy):
        ...     def __init__(self, weight=0.5):
        ...         self.weight = weight
        ...         self._criterion = None
        ...
        ...     def setup(self, context):
        ...         self._criterion = nn.CrossEntropyLoss()
        ...
        ...     def compute_loss(self, outputs, labels, context):
        ...         base_loss = self._criterion(outputs, labels)
        ...         reg_term = self.weight * torch.norm(outputs)
        ...         return base_loss + reg_term
    """

    @abstractmethod
    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute loss given model outputs and labels.

        Args:
            outputs: Model predictions (e.g., logits)
            labels: Ground truth labels
            context: Training context with access to model, device, etc.

        Returns:
            Loss tensor (scalar)

        Note:
            The returned loss should be a scalar tensor that can be
            used for backpropagation.
        """
        pass


class OptimizerStrategy(Strategy):
    """
    Strategy interface for creating and configuring optimizers.

    Implement this interface to customize:
    - Optimizer type (SGD, Adam, etc.)
    - Learning rate and other hyperparameters
    - Parameter groups with different settings
    - Custom optimizer implementations

    Example:
        >>> class MyOptimizerStrategy(OptimizerStrategy):
        ...     def __init__(self, lr=0.01, momentum=0.9):
        ...         self.lr = lr
        ...         self.momentum = momentum
        ...
        ...     def create_optimizer(self, model, context):
        ...         return torch.optim.SGD(
        ...             model.parameters(),
        ...             lr=self.lr,
        ...             momentum=self.momentum
        ...         )
    """

    @abstractmethod
    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """
        Create and return the optimizer for training.

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            Configured optimizer instance

        Note:
            This method is called once at the start of training.
            The optimizer will be used for all training steps.
        """
        pass

    def on_optimizer_step(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> None:
        """
        Hook called after optimizer.step() is executed.

        Use this for:
        - Custom post-step processing
        - Parameter clipping
        - Gradient statistics logging

        Args:
            optimizer: The optimizer that just stepped
            context: Training context
        """
        pass


class TrainingStepStrategy(Strategy):
    """
    Strategy interface for performing forward and backward passes.

    Implement this interface to customize the training step logic:
    - Standard forward-backward-step
    - Gradient accumulation
    - Multiple forward passes (LG-FedAvg)
    - Mixed precision training
    - Gradient clipping

    Example:
        >>> class MyStepStrategy(TrainingStepStrategy):
        ...     def training_step(self, model, optimizer, examples,
        ...                       labels, loss_criterion, context):
        ...         optimizer.zero_grad()
        ...         outputs = model(examples)
        ...         loss = loss_criterion(outputs, labels)
        ...         loss.backward()
        ...         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ...         optimizer.step()
        ...         return loss
    """

    @abstractmethod
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
        Perform one training step (forward + backward + optimize).

        This method encapsulates the core training logic. The typical
        implementation includes:
        1. Zero gradients
        2. Forward pass
        3. Compute loss
        4. Backward pass
        5. Optimizer step

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch (already moved to device)
            labels: Target labels (already moved to device)
            loss_criterion: Callable that computes loss (outputs, labels) -> loss
            context: Training context

        Returns:
            Loss value for this step (for logging/tracking)

        Note:
            The loss_criterion is a lambda that calls the LossCriterionStrategy.
            You can call it as: loss = loss_criterion(outputs, labels)
        """
        pass


class LRSchedulerStrategy(Strategy):
    """
    Strategy interface for learning rate scheduling.

    Implement this interface to customize learning rate scheduling:
    - Step decay
    - Cosine annealing
    - Warmup schedules
    - Custom schedules

    Example:
        >>> class MySchedulerStrategy(LRSchedulerStrategy):
        ...     def __init__(self, step_size=10, gamma=0.1):
        ...         self.step_size = step_size
        ...         self.gamma = gamma
        ...
        ...     def create_scheduler(self, optimizer, context):
        ...         return torch.optim.lr_scheduler.StepLR(
        ...             optimizer,
        ...             step_size=self.step_size,
        ...             gamma=self.gamma
        ...         )
        ...
        ...     def step(self, scheduler, context):
        ...         if scheduler is not None:
        ...             scheduler.step()
    """

    @abstractmethod
    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create and return learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule
            context: Training context

        Returns:
            LR scheduler instance, or None for no scheduling

        Note:
            Return None if no learning rate scheduling is desired.
        """
        pass

    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext,
    ) -> None:
        """
        Perform one scheduler step.

        Args:
            scheduler: The scheduler (may be None)
            context: Training context

        Note:
            This is called after each epoch by default.
            Override for different scheduling intervals.
        """
        if scheduler is not None:
            scheduler.step()


class ModelUpdateStrategy(Strategy):
    """
    Strategy interface for managing model updates and state.

    This is the most flexible strategy interface, used for algorithms
    that need to maintain state or modify training behavior at various
    points in the lifecycle:
    - SCAFFOLD control variates
    - FedDyn state management
    - Personalized FL state handling
    - Custom gradient modifications

    Example:
        >>> class MyUpdateStrategy(ModelUpdateStrategy):
        ...     def __init__(self):
        ...         self.global_weights = None
        ...
        ...     def on_train_start(self, context):
        ...         self.global_weights = copy.deepcopy(
        ...             context.model.state_dict()
        ...         )
        ...
        ...     def after_step(self, context):
        ...         # Apply custom corrections
        ...         with torch.no_grad():
        ...             for param in context.model.parameters():
        ...                 # Your logic here
        ...                 pass
    """

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Called at the start of each training round.

        Use this to:
        - Receive data from server (via context.state)
        - Initialize per-round state
        - Save global model weights
        - Reset counters

        Args:
            context: Training context
        """
        pass

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Called at the end of training round.

        Use this to:
        - Compute final state updates
        - Prepare data for server
        - Save state to disk

        Args:
            context: Training context
        """
        pass

    def before_step(self, context: TrainingContext) -> None:
        """
        Called before each training step.

        Use this for:
        - Pre-step model modifications
        - State updates before forward/backward

        Args:
            context: Training context
        """
        pass

    def after_step(self, context: TrainingContext) -> None:
        """
        Called after each training step.

        Use this for:
        - Post-step model modifications
        - Gradient corrections (SCAFFOLD)
        - State accumulation

        Args:
            context: Training context
        """
        pass

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return additional data to send to server with model update.

        Use this to send:
        - Control variate deltas (SCAFFOLD)
        - State information
        - Metadata

        Args:
            context: Training context

        Returns:
            Dictionary with additional payload data

        Note:
            This data will be merged with the model weights when
            sending updates to the server.
        """
        return {}


class DataLoaderStrategy(Strategy):
    """
    Strategy interface for creating data loaders.

    Implement this interface to customize data loading:
    - Custom batch sampling
    - Data augmentation pipelines
    - Special collate functions
    - Distributed data loading

    Example:
        >>> class MyDataLoaderStrategy(DataLoaderStrategy):
        ...     def __init__(self, num_workers=4):
        ...         self.num_workers = num_workers
        ...
        ...     def create_train_loader(self, trainset, sampler,
        ...                             batch_size, context):
        ...         return torch.utils.data.DataLoader(
        ...             trainset,
        ...             batch_size=batch_size,
        ...             sampler=sampler,
        ...             num_workers=self.num_workers
        ...         )
    """

    @abstractmethod
    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """
        Create training data loader.

        Args:
            trainset: Training dataset
            sampler: Data sampler (indices or Sampler object)
            batch_size: Batch size
            context: Training context

        Returns:
            Configured DataLoader instance
        """
        pass


class TestingStrategy(Strategy):
    """
    Strategy interface for model testing/evaluation.

    Implement this interface to customize testing behavior:
    - Custom evaluation metrics (MIoU, FID, KNN accuracy, etc.)
    - Multi-phase testing (SSL KNN vs personalization)
    - External framework evaluation (HuggingFace, YOLO)
    - Specialized test loops

    Example:
        >>> class MyTestingStrategy(TestingStrategy):
        ...     def test_model(self, model, config, testset, sampler, context):
        ...         model.eval()
        ...         test_loader = torch.utils.data.DataLoader(
        ...             testset, batch_size=config["batch_size"]
        ...         )
        ...         correct = 0
        ...         total = 0
        ...         with torch.no_grad():
        ...             for examples, labels in test_loader:
        ...                 outputs = model(examples)
        ...                 _, predicted = torch.max(outputs, 1)
        ...                 total += labels.size(0)
        ...                 correct += (predicted == labels).sum().item()
        ...         return correct / total
    """

    @abstractmethod
    def test_model(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        """
        Test the model and return accuracy/metric.

        Args:
            model: The model to test
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional data sampler for test set
            context: Training context with device, client_id, etc.

        Returns:
            Accuracy or other evaluation metric (float)

        Note:
            This method should handle moving model to device,
            setting eval mode, and computing the metric.
        """
        pass
