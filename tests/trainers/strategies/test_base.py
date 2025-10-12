"""
Unit tests for base strategy interfaces.

Tests the core strategy interfaces and TrainingContext to ensure
they work correctly and provide the expected API.
"""

import pytest
import torch
import torch.nn as nn

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


class TestTrainingContext:
    """Test suite for TrainingContext."""

    def test_initialization(self):
        """Test that TrainingContext initializes with correct defaults."""
        context = TrainingContext()

        assert context.model is None
        assert context.device is None
        assert context.client_id == 0
        assert context.current_epoch == 0
        assert context.current_round == 0
        assert isinstance(context.config, dict)
        assert len(context.config) == 0
        assert isinstance(context.state, dict)
        assert len(context.state) == 0

    def test_attribute_assignment(self):
        """Test that attributes can be assigned and retrieved."""
        context = TrainingContext()

        # Test model assignment
        model = nn.Linear(10, 2)
        context.model = model
        assert context.model is model

        # Test device assignment
        device = torch.device("cpu")
        context.device = device
        assert context.device == device

        # Test client_id assignment
        context.client_id = 5
        assert context.client_id == 5

        # Test epoch assignment
        context.current_epoch = 3
        assert context.current_epoch == 3

        # Test round assignment
        context.current_round = 10
        assert context.current_round == 10

    def test_config_dict(self):
        """Test that config dictionary works correctly."""
        context = TrainingContext()

        # Add items to config
        context.config["lr"] = 0.01
        context.config["batch_size"] = 32
        context.config["epochs"] = 10

        assert context.config["lr"] == 0.01
        assert context.config["batch_size"] == 32
        assert context.config["epochs"] == 10
        assert len(context.config) == 3

    def test_state_dict(self):
        """Test that state dictionary works for sharing data."""
        context = TrainingContext()

        # Add items to state
        context.state["custom_data"] = [1, 2, 3]
        context.state["tensor"] = torch.tensor([1.0, 2.0, 3.0])
        context.state["nested"] = {"key": "value"}

        assert context.state["custom_data"] == [1, 2, 3]
        assert torch.equal(context.state["tensor"], torch.tensor([1.0, 2.0, 3.0]))
        assert context.state["nested"]["key"] == "value"

    def test_repr(self):
        """Test string representation of context."""
        context = TrainingContext()
        context.client_id = 3
        context.current_epoch = 5
        context.current_round = 2

        repr_str = repr(context)
        assert "TrainingContext" in repr_str
        assert "client_id=3" in repr_str
        assert "epoch=5" in repr_str
        assert "round=2" in repr_str


class TestStrategyBase:
    """Test suite for base Strategy class."""

    def test_strategy_is_abstract(self):
        """Test that Strategy is abstract and has expected methods."""
        # Strategy should have setup and teardown methods
        assert hasattr(Strategy, "setup")
        assert hasattr(Strategy, "teardown")

    def test_strategy_lifecycle_methods(self):
        """Test that lifecycle methods can be called."""

        class ConcreteStrategy(Strategy):
            """Concrete strategy for testing."""

            def __init__(self):
                self.setup_called = False
                self.teardown_called = False

            def setup(self, context):
                self.setup_called = True

            def teardown(self, context):
                self.teardown_called = True

        strategy = ConcreteStrategy()
        context = TrainingContext()

        assert not strategy.setup_called
        assert not strategy.teardown_called

        strategy.setup(context)
        assert strategy.setup_called

        strategy.teardown(context)
        assert strategy.teardown_called


class TestLossCriterionStrategy:
    """Test suite for LossCriterionStrategy interface."""

    def test_interface_has_abstract_method(self):
        """Test that compute_loss is abstract."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            LossCriterionStrategy()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""

        class SimpleLoss(LossCriterionStrategy):
            def __init__(self):
                self._criterion = nn.CrossEntropyLoss()

            def compute_loss(self, outputs, labels, context):
                return self._criterion(outputs, labels)

        strategy = SimpleLoss()
        context = TrainingContext()

        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        loss = strategy.compute_loss(outputs, labels, context)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0


class TestOptimizerStrategy:
    """Test suite for OptimizerStrategy interface."""

    def test_interface_has_abstract_method(self):
        """Test that create_optimizer is abstract."""
        with pytest.raises(TypeError):
            OptimizerStrategy()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""

        class SimpleOptimizer(OptimizerStrategy):
            def __init__(self, lr=0.01):
                self.lr = lr

            def create_optimizer(self, model, context):
                return torch.optim.SGD(model.parameters(), lr=self.lr)

        model = nn.Linear(10, 2)
        context = TrainingContext()
        strategy = SimpleOptimizer(lr=0.05)

        optimizer = strategy.create_optimizer(model, context)

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.param_groups[0]["lr"] == 0.05

    def test_on_optimizer_step_hook(self):
        """Test that on_optimizer_step hook works."""

        class OptimizerWithHook(OptimizerStrategy):
            def __init__(self):
                self.step_count = 0

            def create_optimizer(self, model, context):
                return torch.optim.SGD(model.parameters(), lr=0.01)

            def on_optimizer_step(self, optimizer, context):
                self.step_count += 1

        model = nn.Linear(10, 2)
        context = TrainingContext()
        strategy = OptimizerWithHook()

        optimizer = strategy.create_optimizer(model, context)

        assert strategy.step_count == 0

        # Simulate optimizer step
        strategy.on_optimizer_step(optimizer, context)
        assert strategy.step_count == 1

        strategy.on_optimizer_step(optimizer, context)
        assert strategy.step_count == 2


class TestTrainingStepStrategy:
    """Test suite for TrainingStepStrategy interface."""

    def test_interface_has_abstract_method(self):
        """Test that training_step is abstract."""
        with pytest.raises(TypeError):
            TrainingStepStrategy()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""

        class SimpleStep(TrainingStepStrategy):
            def training_step(
                self, model, optimizer, examples, labels, loss_criterion, context
            ):
                optimizer.zero_grad()
                outputs = model(examples)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                return loss

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        strategy = SimpleStep()
        context = TrainingContext()

        examples = torch.randn(5, 10)
        labels = torch.randint(0, 2, (5,))
        loss_fn = lambda o, l: nn.CrossEntropyLoss()(o, l)

        loss = strategy.training_step(
            model, optimizer, examples, labels, loss_fn, context
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestLRSchedulerStrategy:
    """Test suite for LRSchedulerStrategy interface."""

    def test_interface_has_abstract_method(self):
        """Test that create_scheduler is abstract."""
        with pytest.raises(TypeError):
            LRSchedulerStrategy()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""

        class SimpleScheduler(LRSchedulerStrategy):
            def create_scheduler(self, optimizer, context):
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        context = TrainingContext()
        strategy = SimpleScheduler()

        scheduler = strategy.create_scheduler(optimizer, context)

        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)

    def test_step_method(self):
        """Test that step method works."""

        class SimpleScheduler(LRSchedulerStrategy):
            def __init__(self):
                self.step_count = 0

            def create_scheduler(self, optimizer, context):
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

            def step(self, scheduler, context):
                if scheduler is not None:
                    scheduler.step()
                    self.step_count += 1

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        context = TrainingContext()
        strategy = SimpleScheduler()

        scheduler = strategy.create_scheduler(optimizer, context)
        initial_lr = optimizer.param_groups[0]["lr"]

        strategy.step(scheduler, context)
        assert strategy.step_count == 1

        # LR should have changed (StepLR with step_size=1 and default gamma=0.1)
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_none_scheduler(self):
        """Test that None scheduler is handled gracefully."""

        class NoScheduler(LRSchedulerStrategy):
            def create_scheduler(self, optimizer, context):
                return None

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        context = TrainingContext()
        strategy = NoScheduler()

        scheduler = strategy.create_scheduler(optimizer, context)
        assert scheduler is None

        # Step should not raise error with None scheduler
        strategy.step(scheduler, context)


class TestModelUpdateStrategy:
    """Test suite for ModelUpdateStrategy interface."""

    def test_all_methods_optional(self):
        """Test that all methods are optional (have default implementations)."""

        class MinimalUpdate(ModelUpdateStrategy):
            pass

        strategy = MinimalUpdate()
        context = TrainingContext()

        # All methods should be callable without errors
        strategy.on_train_start(context)
        strategy.on_train_end(context)
        strategy.before_step(context)
        strategy.after_step(context)

        payload = strategy.get_update_payload(context)
        assert isinstance(payload, dict)
        assert len(payload) == 0

    def test_lifecycle_hooks(self):
        """Test that lifecycle hooks work correctly."""

        class TrackingUpdate(ModelUpdateStrategy):
            def __init__(self):
                self.train_started = False
                self.train_ended = False
                self.before_step_count = 0
                self.after_step_count = 0

            def on_train_start(self, context):
                self.train_started = True

            def on_train_end(self, context):
                self.train_ended = True

            def before_step(self, context):
                self.before_step_count += 1

            def after_step(self, context):
                self.after_step_count += 1

        strategy = TrackingUpdate()
        context = TrainingContext()

        assert not strategy.train_started
        assert not strategy.train_ended

        strategy.on_train_start(context)
        assert strategy.train_started

        strategy.before_step(context)
        strategy.before_step(context)
        assert strategy.before_step_count == 2

        strategy.after_step(context)
        assert strategy.after_step_count == 1

        strategy.on_train_end(context)
        assert strategy.train_ended

    def test_get_update_payload(self):
        """Test that get_update_payload returns data."""

        class PayloadUpdate(ModelUpdateStrategy):
            def __init__(self):
                self.step_count = 0

            def after_step(self, context):
                self.step_count += 1
                context.state["steps"] = self.step_count

            def get_update_payload(self, context):
                return {"step_count": self.step_count}

        strategy = PayloadUpdate()
        context = TrainingContext()

        strategy.after_step(context)
        strategy.after_step(context)

        payload = strategy.get_update_payload(context)
        assert payload["step_count"] == 2
        assert context.state["steps"] == 2


class TestDataLoaderStrategy:
    """Test suite for DataLoaderStrategy interface."""

    def test_interface_has_abstract_method(self):
        """Test that create_train_loader is abstract."""
        with pytest.raises(TypeError):
            DataLoaderStrategy()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""

        class SimpleDataLoader(DataLoaderStrategy):
            def create_train_loader(self, trainset, sampler, batch_size, context):
                return torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, shuffle=False
                )

        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 2, (100,))
        )

        context = TrainingContext()
        strategy = SimpleDataLoader()

        loader = strategy.create_train_loader(dataset, None, 32, context)

        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 32

        # Check that data can be loaded
        for batch_x, batch_y in loader:
            assert batch_x.shape[0] <= 32
            assert batch_y.shape[0] <= 32
            break  # Just check first batch


class TestStrategyComposition:
    """Test that strategies can work together."""

    def test_context_sharing_between_strategies(self):
        """Test that multiple strategies can share data via context."""

        class LossStrategy(LossCriterionStrategy):
            def compute_loss(self, outputs, labels, context):
                loss = nn.CrossEntropyLoss()(outputs, labels)
                context.state["last_loss"] = loss.item()
                return loss

        class UpdateStrategy(ModelUpdateStrategy):
            def on_train_end(self, context):
                # Access loss stored by loss strategy
                context.state["avg_loss"] = context.state.get("last_loss", 0.0)

        context = TrainingContext()
        context.model = nn.Linear(10, 2)

        loss_strategy = LossStrategy()
        update_strategy = UpdateStrategy()

        # Simulate training
        outputs = torch.randn(5, 2)
        labels = torch.randint(0, 2, (5,))

        loss = loss_strategy.compute_loss(outputs, labels, context)
        assert "last_loss" in context.state

        update_strategy.on_train_end(context)
        assert "avg_loss" in context.state
        assert context.state["avg_loss"] == context.state["last_loss"]
