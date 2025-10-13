"""
Trainer strategies for composable architecture.

This package provides strategy interfaces and implementations for the
composable trainer design pattern. Strategies allow customizing different
aspects of training through composition instead of inheritance.

Strategy Types:
    - LossCriterionStrategy: Customize loss computation
    - OptimizerStrategy: Customize optimizer creation
    - TrainingStepStrategy: Customize training step logic
    - LRSchedulerStrategy: Customize learning rate scheduling
    - ModelUpdateStrategy: Manage state and model updates
    - DataLoaderStrategy: Customize data loading

Example:
    >>> from plato.trainers.composable import ComposableTrainer
    >>> from plato.trainers.strategies import (
    ...     CrossEntropyLossStrategy,
    ...     AdamOptimizerStrategy,
    ... )
    >>>
    >>> trainer = ComposableTrainer(
    ...     loss_strategy=CrossEntropyLossStrategy(),
    ...     optimizer_strategy=AdamOptimizerStrategy(lr=0.001)
    ... )
"""

# Base classes
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    LossCriterionStrategy,
    LRSchedulerStrategy,
    ModelUpdateStrategy,
    OptimizerStrategy,
    Strategy,
    TrainingContext,
    TrainingStepStrategy,
)

# Data loader strategies
from plato.trainers.strategies.data_loader import (
    CustomCollateFnDataLoaderStrategy,
    DefaultDataLoaderStrategy,
    DynamicBatchSizeDataLoaderStrategy,
    PrefetchDataLoaderStrategy,
    ShuffleDataLoaderStrategy,
)

# Loss criterion strategies
from plato.trainers.strategies.loss_criterion import (
    BCEWithLogitsLossStrategy,
    CompositeLossStrategy,
    CrossEntropyLossStrategy,
    DefaultLossCriterionStrategy,
    L2RegularizationStrategy,
    MSELossStrategy,
    NLLLossStrategy,
)

# LR scheduler strategies
from plato.trainers.strategies.lr_scheduler import (
    CosineAnnealingLRSchedulerStrategy,
    CosineAnnealingWarmRestartsSchedulerStrategy,
    DefaultLRSchedulerStrategy,
    ExponentialLRSchedulerStrategy,
    LinearLRSchedulerStrategy,
    MultiStepLRSchedulerStrategy,
    NoSchedulerStrategy,
    PolynomialLRSchedulerStrategy,
    ReduceLROnPlateauSchedulerStrategy,
    StepLRSchedulerStrategy,
    TimmLRSchedulerStrategy,
    WarmupSchedulerStrategy,
)

# Model update strategies
from plato.trainers.strategies.model_update import (
    CompositeUpdateStrategy,
    NoOpUpdateStrategy,
    StateTrackingUpdateStrategy,
)

# Optimizer strategies
from plato.trainers.strategies.optimizer import (
    AdamOptimizerStrategy,
    AdamWOptimizerStrategy,
    DefaultOptimizerStrategy,
    GradientClippingOptimizerStrategy,
    ParameterGroupOptimizerStrategy,
    RMSpropOptimizerStrategy,
    SGDOptimizerStrategy,
)

# Training step strategies
from plato.trainers.strategies.training_step import (
    CustomBackwardStepStrategy,
    DefaultTrainingStepStrategy,
    GradientAccumulationStepStrategy,
    GradientClippingStepStrategy,
    MixedPrecisionStepStrategy,
    MultipleForwardPassStepStrategy,
    ValidateBeforeStepStrategy,
)

__all__ = [
    # Base
    "Strategy",
    "TrainingContext",
    "LossCriterionStrategy",
    "OptimizerStrategy",
    "TrainingStepStrategy",
    "LRSchedulerStrategy",
    "ModelUpdateStrategy",
    "DataLoaderStrategy",
    # Loss criterion
    "DefaultLossCriterionStrategy",
    "CrossEntropyLossStrategy",
    "MSELossStrategy",
    "BCEWithLogitsLossStrategy",
    "NLLLossStrategy",
    "CompositeLossStrategy",
    "L2RegularizationStrategy",
    # Optimizer
    "DefaultOptimizerStrategy",
    "SGDOptimizerStrategy",
    "AdamOptimizerStrategy",
    "AdamWOptimizerStrategy",
    "RMSpropOptimizerStrategy",
    "ParameterGroupOptimizerStrategy",
    "GradientClippingOptimizerStrategy",
    # Training step
    "DefaultTrainingStepStrategy",
    "GradientAccumulationStepStrategy",
    "MixedPrecisionStepStrategy",
    "GradientClippingStepStrategy",
    "CustomBackwardStepStrategy",
    "MultipleForwardPassStepStrategy",
    "ValidateBeforeStepStrategy",
    # LR scheduler
    "DefaultLRSchedulerStrategy",
    "NoSchedulerStrategy",
    "StepLRSchedulerStrategy",
    "MultiStepLRSchedulerStrategy",
    "ExponentialLRSchedulerStrategy",
    "CosineAnnealingLRSchedulerStrategy",
    "CosineAnnealingWarmRestartsSchedulerStrategy",
    "ReduceLROnPlateauSchedulerStrategy",
    "LinearLRSchedulerStrategy",
    "PolynomialLRSchedulerStrategy",
    "TimmLRSchedulerStrategy",
    "WarmupSchedulerStrategy",
    # Model update
    "NoOpUpdateStrategy",
    "StateTrackingUpdateStrategy",
    "CompositeUpdateStrategy",
    # Data loader
    "DefaultDataLoaderStrategy",
    "CustomCollateFnDataLoaderStrategy",
    "PrefetchDataLoaderStrategy",
    "DynamicBatchSizeDataLoaderStrategy",
    "ShuffleDataLoaderStrategy",
]
