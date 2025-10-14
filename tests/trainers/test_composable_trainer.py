"""
Integration tests for ComposableTrainer.

Tests the ComposableTrainer with various strategy combinations to ensure
it works correctly in end-to-end training scenarios.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    AdamOptimizerStrategy,
    CrossEntropyLossStrategy,
    DefaultDataLoaderStrategy,
    DefaultTrainingStepStrategy,
    NoOpUpdateStrategy,
    NoSchedulerStrategy,
)
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    ModelUpdateStrategy,
    TrainingContext,
)


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set up environment variables for testing."""
    import os
    import sys

    test_args = ["pytest"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setenv("config_file", "tests/config.yml")
    return None


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return lambda: nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def simple_config():
    """Create a simple training config."""
    return {
        "batch_size": 16,
        "epochs": 2,
        "lr": 0.01,
        "run_id": "test",
    }


class TestComposableTrainerBasic:
    """Test basic ComposableTrainer functionality."""

    def test_initialization_with_defaults(self, simple_model):
        """Test that trainer initializes with all default strategies."""
        trainer = ComposableTrainer(model=simple_model)

        assert trainer.model is not None
        assert trainer.loss_strategy is not None
        assert trainer.optimizer_strategy is not None
        assert trainer.training_step_strategy is not None
        assert trainer.lr_scheduler_strategy is not None
        assert trainer.model_update_strategy is not None
        assert trainer.data_loader_strategy is not None

    def test_initialization_with_custom_strategies(self, simple_model):
        """Test that trainer accepts custom strategies."""
        loss_strategy = CrossEntropyLossStrategy()
        optimizer_strategy = AdamOptimizerStrategy(lr=0.001)

        trainer = ComposableTrainer(
            model=simple_model,
            loss_strategy=loss_strategy,
            optimizer_strategy=optimizer_strategy,
        )

        assert trainer.loss_strategy is loss_strategy
        assert trainer.optimizer_strategy is optimizer_strategy

    def test_context_initialization(self, simple_model):
        """Test that training context is initialized correctly."""
        trainer = ComposableTrainer(model=simple_model)

        assert trainer.context is not None
        assert trainer.context.model is trainer.model
        assert trainer.context.device is not None
        assert trainer.context.client_id == 0

    def test_strategies_are_setup(self, simple_model):
        """Test that all strategies are setup during initialization."""

        class TrackingLossStrategy(CrossEntropyLossStrategy):
            def __init__(self):
                super().__init__()
                self.setup_called = False

            def setup(self, context):
                super().setup(context)
                self.setup_called = True

        loss_strategy = TrackingLossStrategy()
        trainer = ComposableTrainer(model=simple_model, loss_strategy=loss_strategy)

        assert loss_strategy.setup_called


class TestComposableTrainerTraining:
    """Test training functionality."""

    def test_train_model_executes(self, simple_model, simple_dataset, simple_config):
        """Test that train_model executes without errors."""
        trainer = ComposableTrainer(model=simple_model)
        sampler = list(range(len(simple_dataset)))

        # Should not raise any exceptions
        trainer.train_model(simple_config, simple_dataset, sampler)

        # Verify training occurred
        assert trainer.current_epoch == 2
        assert len(trainer.run_history.get_metric_values("train_loss")) > 0

    def test_loss_decreases_during_training(
        self, simple_model, simple_dataset, simple_config
    ):
        """Test that loss generally decreases during training."""
        trainer = ComposableTrainer(
            model=simple_model,
            loss_strategy=CrossEntropyLossStrategy(),
            optimizer_strategy=AdamOptimizerStrategy(lr=0.01),
        )
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        # Get loss history
        loss_history = trainer.run_history.get_metric_values("train_loss")

        assert len(loss_history) > 0
        # Loss should decrease (first loss > last loss)
        assert loss_history[0] > loss_history[-1]

    def test_multiple_epochs(self, simple_model, simple_dataset):
        """Test training for multiple epochs."""
        config = {
            "batch_size": 16,
            "epochs": 5,
            "lr": 0.01,
            "run_id": "test",
        }

        trainer = ComposableTrainer(model=simple_model)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(config, simple_dataset, sampler)

        assert trainer.current_epoch == 5
        assert len(trainer.run_history.get_metric_values("train_loss")) == 5


class TestComposableTrainerStrategies:
    """Test strategy integration."""

    def test_loss_strategy_is_used(self, simple_model, simple_dataset, simple_config):
        """Test that custom loss strategy is actually used."""

        class CountingLossStrategy(LossCriterionStrategy):
            def __init__(self):
                self.call_count = 0
                self._criterion = nn.CrossEntropyLoss()

            def setup(self, context):
                pass

            def compute_loss(self, outputs, labels, context):
                self.call_count += 1
                return self._criterion(outputs, labels)

        loss_strategy = CountingLossStrategy()
        trainer = ComposableTrainer(model=simple_model, loss_strategy=loss_strategy)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        # Loss should have been computed multiple times
        assert loss_strategy.call_count > 0

    def test_model_update_strategy_hooks(
        self, simple_model, simple_dataset, simple_config
    ):
        """Test that model update strategy hooks are called."""

        class TrackingUpdateStrategy(ModelUpdateStrategy):
            def __init__(self):
                self.on_train_start_called = False
                self.on_train_end_called = False
                self.before_step_count = 0
                self.after_step_count = 0

            def on_train_start(self, context):
                self.on_train_start_called = True

            def on_train_end(self, context):
                self.on_train_end_called = True

            def before_step(self, context):
                self.before_step_count += 1

            def after_step(self, context):
                self.after_step_count += 1

        update_strategy = TrackingUpdateStrategy()
        trainer = ComposableTrainer(
            model=simple_model, model_update_strategy=update_strategy
        )
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        assert update_strategy.on_train_start_called
        assert update_strategy.on_train_end_called
        assert update_strategy.before_step_count > 0
        assert update_strategy.after_step_count > 0

    def test_optimizer_strategy_creates_optimizer(
        self, simple_model, simple_dataset, simple_config
    ):
        """Test that optimizer strategy creates the correct optimizer."""
        optimizer_strategy = AdamOptimizerStrategy(lr=0.005)
        trainer = ComposableTrainer(
            model=simple_model, optimizer_strategy=optimizer_strategy
        )
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.005


class TestComposableTrainerContext:
    """Test context sharing between strategies."""

    def test_context_shared_between_strategies(
        self, simple_model, simple_dataset, simple_config
    ):
        """Test that strategies can share data via context."""

        class WriterStrategy(LossCriterionStrategy):
            def __init__(self):
                self._criterion = nn.CrossEntropyLoss()

            def compute_loss(self, outputs, labels, context):
                loss = self._criterion(outputs, labels)
                context.state["shared_data"] = loss.item()
                return loss

        class ReaderStrategy(ModelUpdateStrategy):
            def __init__(self):
                self.read_data = None

            def on_train_end(self, context):
                self.read_data = context.state.get("shared_data")

        writer = WriterStrategy()
        reader = ReaderStrategy()

        trainer = ComposableTrainer(
            model=simple_model, loss_strategy=writer, model_update_strategy=reader
        )
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        # Reader should have received data from writer
        assert reader.read_data is not None

    def test_context_updates_during_training(
        self, simple_model, simple_dataset, simple_config
    ):
        """Test that context is updated during training."""

        class ContextCheckStrategy(ModelUpdateStrategy):
            def __init__(self):
                self.epochs_seen = set()
                self.batches_seen = set()

            def after_step(self, context):
                self.epochs_seen.add(context.current_epoch)
                self.batches_seen.add(context.state.get("current_batch"))

        strategy = ContextCheckStrategy()
        trainer = ComposableTrainer(model=simple_model, model_update_strategy=strategy)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        # Should have seen multiple epochs and batches
        assert len(strategy.epochs_seen) == simple_config["epochs"]
        assert len(strategy.batches_seen) > 0


class TestComposableTrainerCallbacks:
    """Test callback integration."""

    def test_callbacks_are_called(self, simple_model, simple_dataset, simple_config):
        """Test that callbacks work with composable trainer."""
        from plato.callbacks.trainer import TrainerCallback

        class TrackingCallback(TrainerCallback):
            def __init__(self):
                self.train_run_start_called = False
                self.train_epoch_start_called = False
                self.train_step_end_count = 0

            def on_train_run_start(self, trainer, config, **kwargs):
                self.train_run_start_called = True

            def on_train_epoch_start(self, trainer, config, **kwargs):
                self.train_epoch_start_called = True

            def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
                self.train_step_end_count += 1

        callback = TrackingCallback()
        trainer = ComposableTrainer(model=simple_model, callbacks=[callback])
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        assert callback.train_run_start_called
        assert callback.train_epoch_start_called
        assert callback.train_step_end_count > 0


class TestComposableTrainerModelOperations:
    """Test model save/load operations."""

    def test_save_and_load_model(self, simple_model, tmp_path):
        """Test that model can be saved and loaded."""
        trainer = ComposableTrainer(model=simple_model)

        # Create some dummy weights
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.fill_(1.0)

        # Save model
        trainer.save_model(filename="test_model.pth", location=str(tmp_path))

        # Create new trainer and load
        trainer2 = ComposableTrainer(model=simple_model)
        trainer2.load_model(filename="test_model.pth", location=str(tmp_path))

        # Check that weights match
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)


class TestComposableTrainerWithDifferentModels:
    """Test trainer with different model types."""

    def test_with_callable_model(self, simple_dataset, simple_config):
        """Test trainer with callable model."""

        def create_model():
            return nn.Linear(10, 2)

        trainer = ComposableTrainer(model=create_model)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        assert trainer.model is not None
        assert isinstance(trainer.model, nn.Linear)

    def test_with_model_instance(self, simple_dataset, simple_config):
        """Test trainer with model instance."""
        model_instance = nn.Linear(10, 2)
        trainer = ComposableTrainer(model=model_instance)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        assert trainer.model is model_instance


class TestComposableTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, simple_model, simple_config):
        """Test behavior with empty dataset."""
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randint(0, 2, (0,)))
        sampler = list(range(len(empty_dataset)))

        trainer = ComposableTrainer(model=simple_model)

        # Should handle gracefully (no batches to process)
        trainer.train_model(simple_config, empty_dataset, sampler)

        # Should complete without errors
        assert trainer.current_epoch == simple_config["epochs"]

    def test_single_batch(self, simple_model, simple_config):
        """Test training with single batch."""
        small_dataset = TensorDataset(torch.randn(5, 10), torch.randint(0, 2, (5,)))
        sampler = list(range(len(small_dataset)))

        config = simple_config.copy()
        config["batch_size"] = 10  # Larger than dataset

        trainer = ComposableTrainer(model=simple_model)
        trainer.train_model(config, small_dataset, sampler)

        # Should complete successfully
        assert trainer.current_epoch == config["epochs"]


class TestComposableTrainerComparison:
    """Compare ComposableTrainer with default strategies to basic behavior."""

    def test_produces_valid_loss(self, simple_model, simple_dataset, simple_config):
        """Test that training produces valid (finite) loss values."""
        trainer = ComposableTrainer(model=simple_model)
        sampler = list(range(len(simple_dataset)))

        trainer.train_model(simple_config, simple_dataset, sampler)

        loss_history = trainer.run_history.get_metric_values("train_loss")

        # All losses should be finite
        for loss in loss_history:
            assert torch.isfinite(torch.tensor(loss))
            assert loss > 0  # Loss should be positive

    def test_model_parameters_change(self, simple_model, simple_dataset, simple_config):
        """Test that model parameters actually change during training."""
        trainer = ComposableTrainer(model=simple_model)

        # Save initial parameters (move to same device as they'll be after training)
        initial_params = [
            p.clone().to(trainer.context.device) for p in trainer.model.parameters()
        ]

        sampler = list(range(len(simple_dataset)))
        trainer.train_model(simple_config, simple_dataset, sampler)

        # Check that at least some parameters changed
        params_changed = False
        for initial, final in zip(initial_params, trainer.model.parameters()):
            if not torch.allclose(initial, final):
                params_changed = True
                break

        assert params_changed, "Model parameters should change during training"
