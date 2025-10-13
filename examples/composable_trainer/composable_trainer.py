"""
Example demonstrating the ComposableTrainer with strategy composition.

This example shows how to use the new composable trainer architecture
to customize different aspects of training through strategy injection
instead of inheritance.

The example demonstrates:
1. Using default strategies (simplest approach)
2. Customizing individual strategies
3. Combining multiple custom strategies
4. Creating custom strategies

Run this example:
    uv run composable_trainer_example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    AdamOptimizerStrategy,
    CosineAnnealingLRSchedulerStrategy,
    CrossEntropyLossStrategy,
    DefaultDataLoaderStrategy,
    GradientAccumulationStepStrategy,
    MixedPrecisionStepStrategy,
)
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


def create_simple_model():
    """Create a simple neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def create_simple_dataset(n_samples=1000):
    """Create a simple synthetic dataset for demonstration."""
    # Generate random features
    X = torch.randn(n_samples, 10)

    # Generate labels (simple linear separation)
    weights = torch.randn(10)
    y = (X @ weights > 0).long()

    return TensorDataset(X, y)


def example_1_default_strategies():
    """
    Example 1: Using all default strategies.

    This is the simplest approach - just provide a model and let
    the trainer use default strategies for everything.
    """
    print("\n" + "=" * 70)
    print("Example 1: Using Default Strategies")
    print("=" * 70)

    # Create trainer with all defaults
    trainer = ComposableTrainer(model=create_simple_model)

    # Create dataset and config
    dataset = create_simple_dataset(n_samples=500)
    sampler = list(range(len(dataset)))
    config = {
        "batch_size": 32,
        "epochs": 3,
        "lr": 0.01,
        "run_id": "example1",
    }

    # Train
    print("Training with default strategies...")
    trainer.train_model(config, dataset, sampler)

    # Show results
    loss_history = trainer.run_history.get_metric_values("train_loss")
    print(f"\nTraining complete!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Loss reduction: {(1 - loss_history[-1] / loss_history[0]) * 100:.1f}%")


def example_2_custom_loss_and_optimizer():
    """
    Example 2: Customize loss and optimizer strategies.

    This shows how to inject custom strategies for specific components
    while using defaults for others.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Loss and Optimizer Strategies")
    print("=" * 70)

    # Create trainer with custom strategies
    trainer = ComposableTrainer(
        model=create_simple_model,
        loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1),
        optimizer_strategy=AdamOptimizerStrategy(
            lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
        ),
    )

    # Create dataset and config
    dataset = create_simple_dataset(n_samples=500)
    sampler = list(range(len(dataset)))
    config = {
        "batch_size": 32,
        "epochs": 3,
        "lr": 0.001,  # This will be used by optimizer strategy
        "run_id": "example2",
    }

    # Train
    print("Training with CrossEntropy (label_smoothing=0.1) and Adam optimizer...")
    trainer.train_model(config, dataset, sampler)

    # Show results
    loss_history = trainer.run_history.get_metric_values("train_loss")
    print(f"\nTraining complete!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Optimizer: {type(trainer.optimizer).__name__}")


def example_3_multiple_strategies():
    """
    Example 3: Combine multiple custom strategies.

    This demonstrates the power of composition - mixing different
    strategies to create complex training configurations.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multiple Custom Strategies")
    print("=" * 70)

    # Create trainer with multiple custom strategies
    trainer = ComposableTrainer(
        model=create_simple_model,
        loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.05),
        optimizer_strategy=AdamOptimizerStrategy(lr=0.001, weight_decay=0.01),
        training_step_strategy=GradientAccumulationStepStrategy(
            accumulation_steps=4  # Effectively 4x batch size
        ),
        lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=3),
        data_loader_strategy=DefaultDataLoaderStrategy(num_workers=0),
    )

    # Create dataset and config
    dataset = create_simple_dataset(n_samples=500)
    sampler = list(range(len(dataset)))
    config = {
        "batch_size": 16,  # Small batch, but accumulation makes it effectively 64
        "epochs": 3,
        "lr": 0.001,
        "run_id": "example3",
    }

    # Train
    print("Training with:")
    print("  - CrossEntropy loss with label smoothing")
    print("  - Adam optimizer with weight decay")
    print("  - Gradient accumulation (4 steps)")
    print("  - Cosine annealing LR scheduler")
    trainer.train_model(config, dataset, sampler)

    # Show results
    loss_history = trainer.run_history.get_metric_values("train_loss")
    print(f"\nTraining complete!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Effective batch size: {config['batch_size'] * 4}")


def example_4_custom_strategy():
    """
    Example 4: Create and use a custom strategy.

    This shows how to implement your own strategy by inheriting from
    the base strategy class.
    """
    print("\n" + "=" * 70)
    print("Example 4: Custom Strategy Implementation")
    print("=" * 70)

    # Define a custom loss strategy
    class CustomWeightedLossStrategy(LossCriterionStrategy):
        """
        Custom loss that weights classes differently and adds L2 regularization.
        """

        def __init__(self, class_weights, l2_weight=0.01):
            self.class_weights = class_weights
            self.l2_weight = l2_weight
            self._criterion = None

        def setup(self, context: TrainingContext):
            """Initialize the loss criterion."""
            weights = torch.tensor(self.class_weights).to(context.device)
            self._criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Custom loss initialized with class weights: {self.class_weights}")
            print(f"L2 regularization weight: {self.l2_weight}")

        def compute_loss(self, outputs, labels, context):
            """Compute weighted cross-entropy loss + L2 regularization."""
            # Base cross-entropy loss with class weights
            ce_loss = self._criterion(outputs, labels)

            # Add L2 regularization
            l2_reg = torch.tensor(0.0, device=outputs.device)
            for param in context.model.parameters():
                l2_reg += torch.sum(param**2)

            total_loss = ce_loss + self.l2_weight * l2_reg

            return total_loss

    # Create trainer with custom strategy
    trainer = ComposableTrainer(
        model=create_simple_model,
        loss_strategy=CustomWeightedLossStrategy(
            class_weights=[1.0, 2.0],  # Weight class 1 more heavily
            l2_weight=0.001,
        ),
        optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    )

    # Create dataset and config
    dataset = create_simple_dataset(n_samples=500)
    sampler = list(range(len(dataset)))
    config = {
        "batch_size": 32,
        "epochs": 3,
        "lr": 0.001,
        "run_id": "example4",
    }

    # Train
    print("\nTraining with custom weighted loss + L2 regularization...")
    trainer.train_model(config, dataset, sampler)

    # Show results
    loss_history = trainer.run_history.get_metric_values("train_loss")
    print(f"\nTraining complete!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")


def example_5_mixed_precision():
    """
    Example 5: Mixed precision training for faster computation.

    This demonstrates using automatic mixed precision (AMP) for
    faster training on compatible hardware.
    """
    print("\n" + "=" * 70)
    print("Example 5: Mixed Precision Training")
    print("=" * 70)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA available - using mixed precision training")
        enabled = True
    else:
        print("CUDA not available - mixed precision disabled")
        enabled = False

    # Create trainer with mixed precision
    trainer = ComposableTrainer(
        model=create_simple_model,
        training_step_strategy=MixedPrecisionStepStrategy(enabled=enabled),
        optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    )

    # Create dataset and config
    dataset = create_simple_dataset(n_samples=500)
    sampler = list(range(len(dataset)))
    config = {
        "batch_size": 32,
        "epochs": 3,
        "lr": 0.001,
        "run_id": "example5",
    }

    # Train
    print("Training with automatic mixed precision...")
    trainer.train_model(config, dataset, sampler)

    # Show results
    loss_history = trainer.run_history.get_metric_values("train_loss")
    print(f"\nTraining complete!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ComposableTrainer Examples")
    print("=" * 70)
    print("\nThis script demonstrates various ways to use the ComposableTrainer")
    print("with different strategy combinations.\n")

    # Run examples
    example_1_default_strategies()
    example_2_custom_loss_and_optimizer()
    example_3_multiple_strategies()
    example_4_custom_strategy()
    example_5_mixed_precision()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. ComposableTrainer works with all default strategies")
    print("2. Individual strategies can be customized independently")
    print("3. Multiple strategies can be combined easily")
    print("4. Custom strategies are straightforward to implement")
    print("5. Advanced features like mixed precision are simple to enable")
    print("\nFor more information, see the Plato documentation.")
    print()


if __name__ == "__main__":
    main()
