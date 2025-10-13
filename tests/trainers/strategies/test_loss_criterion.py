"""
Unit tests for loss criterion strategies.

Tests the default loss criterion strategy implementations to ensure
they compute losses correctly.
"""

import pytest
import torch
import torch.nn as nn

from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.loss_criterion import (
    BCEWithLogitsLossStrategy,
    CompositeLossStrategy,
    CrossEntropyLossStrategy,
    DefaultLossCriterionStrategy,
    L2RegularizationStrategy,
    MSELossStrategy,
    NLLLossStrategy,
)


class TestCrossEntropyLossStrategy:
    """Test suite for CrossEntropyLossStrategy."""

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = CrossEntropyLossStrategy(label_smoothing=0.1)
        assert strategy.label_smoothing == 0.1
        assert strategy._criterion is None

    def test_setup(self):
        """Test that setup creates the loss criterion."""
        strategy = CrossEntropyLossStrategy()
        context = TrainingContext()

        assert strategy._criterion is None
        strategy.setup(context)
        assert strategy._criterion is not None
        assert isinstance(strategy._criterion, nn.CrossEntropyLoss)

    def test_compute_loss(self):
        """Test that loss is computed correctly."""
        strategy = CrossEntropyLossStrategy()
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        loss = strategy.compute_loss(outputs, labels, context)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_label_smoothing(self):
        """Test that label smoothing affects loss value."""
        strategy_no_smoothing = CrossEntropyLossStrategy(label_smoothing=0.0)
        strategy_with_smoothing = CrossEntropyLossStrategy(label_smoothing=0.1)

        context = TrainingContext()
        strategy_no_smoothing.setup(context)
        strategy_with_smoothing.setup(context)

        # Use same inputs
        torch.manual_seed(42)
        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        loss_no_smoothing = strategy_no_smoothing.compute_loss(outputs, labels, context)
        loss_with_smoothing = strategy_with_smoothing.compute_loss(
            outputs, labels, context
        )

        # Losses should be different
        assert loss_no_smoothing != loss_with_smoothing

    def test_with_weights(self):
        """Test that class weights work."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        strategy = CrossEntropyLossStrategy(weight=weights)
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        loss = strategy.compute_loss(outputs, labels, context)
        assert loss.item() > 0


class TestMSELossStrategy:
    """Test suite for MSELossStrategy."""

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = MSELossStrategy(reduction="mean")
        assert strategy.reduction == "mean"

    def test_compute_loss(self):
        """Test that MSE loss is computed correctly."""
        strategy = MSELossStrategy()
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.randn(10, 1)
        labels = torch.randn(10, 1)

        loss = strategy.compute_loss(outputs, labels, context)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0  # MSE is always non-negative

    def test_perfect_prediction(self):
        """Test that MSE is zero for perfect predictions."""
        strategy = MSELossStrategy()
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([[1.0], [2.0], [3.0]])

        loss = strategy.compute_loss(outputs, labels, context)

        assert loss.item() < 1e-6  # Should be very close to zero


class TestBCEWithLogitsLossStrategy:
    """Test suite for BCEWithLogitsLossStrategy."""

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = BCEWithLogitsLossStrategy(reduction="sum")
        assert strategy.reduction == "sum"

    def test_compute_loss(self):
        """Test that BCE loss is computed correctly."""
        strategy = BCEWithLogitsLossStrategy()
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.randn(10, 1)
        labels = torch.randint(0, 2, (10, 1)).float()

        loss = strategy.compute_loss(outputs, labels, context)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_with_pos_weight(self):
        """Test that positive class weighting works."""
        pos_weight = torch.tensor([2.0])
        strategy = BCEWithLogitsLossStrategy(pos_weight=pos_weight)
        context = TrainingContext()
        strategy.setup(context)

        outputs = torch.randn(10, 1)
        labels = torch.ones(10, 1)  # All positive class

        loss = strategy.compute_loss(outputs, labels, context)
        assert loss.item() > 0


class TestNLLLossStrategy:
    """Test suite for NLLLossStrategy."""

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = NLLLossStrategy(reduction="mean")
        assert strategy.reduction == "mean"

    def test_compute_loss(self):
        """Test that NLL loss is computed correctly."""
        strategy = NLLLossStrategy()
        context = TrainingContext()
        strategy.setup(context)

        # NLL expects log-probabilities as input
        log_probs = torch.log_softmax(torch.randn(10, 3), dim=1)
        labels = torch.randint(0, 3, (10,))

        loss = strategy.compute_loss(log_probs, labels, context)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestCompositeLossStrategy:
    """Test suite for CompositeLossStrategy."""

    def test_initialization_with_weights(self):
        """Test initialization with weighted strategies."""
        strategy1 = CrossEntropyLossStrategy()
        strategy2 = L2RegularizationStrategy(weight=0.01)

        composite = CompositeLossStrategy([(strategy1, 1.0), (strategy2, 0.5)])

        assert len(composite.strategies) == 2
        assert composite.strategies[0][1] == 1.0
        assert composite.strategies[1][1] == 0.5

    def test_initialization_without_weights(self):
        """Test initialization with default weights."""
        strategy1 = CrossEntropyLossStrategy()
        strategy2 = L2RegularizationStrategy(weight=0.01)

        composite = CompositeLossStrategy([strategy1, strategy2])

        assert len(composite.strategies) == 2
        assert composite.strategies[0][1] == 1.0
        assert composite.strategies[1][1] == 1.0

    def test_setup_calls_all_strategies(self):
        """Test that setup is called on all component strategies."""

        class TrackingLoss(CrossEntropyLossStrategy):
            def __init__(self):
                super().__init__()
                self.setup_called = False

            def setup(self, context):
                super().setup(context)
                self.setup_called = True

        strategy1 = TrackingLoss()
        strategy2 = TrackingLoss()

        composite = CompositeLossStrategy([strategy1, strategy2])
        context = TrainingContext()

        assert not strategy1.setup_called
        assert not strategy2.setup_called

        composite.setup(context)

        assert strategy1.setup_called
        assert strategy2.setup_called

    def test_compute_loss_combines_strategies(self):
        """Test that losses are combined correctly."""
        strategy1 = CrossEntropyLossStrategy()
        strategy2 = CrossEntropyLossStrategy()

        composite = CompositeLossStrategy([(strategy1, 1.0), (strategy2, 0.5)])

        context = TrainingContext()
        context.model = nn.Linear(10, 3)
        composite.setup(context)

        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        # Compute individual losses
        loss1 = strategy1.compute_loss(outputs, labels, context)
        loss2 = strategy2.compute_loss(outputs, labels, context)

        # Compute composite loss
        composite_loss = composite.compute_loss(outputs, labels, context)

        # Should be weighted sum
        expected = 1.0 * loss1 + 0.5 * loss2
        assert torch.allclose(composite_loss, expected, atol=1e-6)

    def test_teardown_calls_all_strategies(self):
        """Test that teardown is called on all component strategies."""

        class TrackingLoss(CrossEntropyLossStrategy):
            def __init__(self):
                super().__init__()
                self.teardown_called = False

            def teardown(self, context):
                super().teardown(context)
                self.teardown_called = True

        strategy1 = TrackingLoss()
        strategy2 = TrackingLoss()

        composite = CompositeLossStrategy([strategy1, strategy2])
        context = TrainingContext()

        assert not strategy1.teardown_called
        assert not strategy2.teardown_called

        composite.teardown(context)

        assert strategy1.teardown_called
        assert strategy2.teardown_called


class TestL2RegularizationStrategy:
    """Test suite for L2RegularizationStrategy."""

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = L2RegularizationStrategy(weight=0.01)
        assert strategy.weight == 0.01

    def test_compute_loss(self):
        """Test that L2 regularization is computed correctly."""
        strategy = L2RegularizationStrategy(weight=0.5)
        context = TrainingContext()

        # Create model with known parameters
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        context.model = model

        outputs = torch.randn(5, 1)
        labels = torch.randn(5, 1)

        loss = strategy.compute_loss(outputs, labels, context)

        # L2 loss = weight * sum(param^2)
        # For our model: weight has 2 elements (all 1), bias has 1 element (all 1)
        # Total: 0.5 * (2*1^2 + 1*1^2) = 0.5 * 3 = 1.5
        expected = 0.5 * 3.0
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)

    def test_zero_parameters(self):
        """Test that L2 loss is zero for zero parameters."""
        strategy = L2RegularizationStrategy(weight=0.5)
        context = TrainingContext()

        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(0.0)
            model.bias.fill_(0.0)

        context.model = model

        outputs = torch.randn(5, 1)
        labels = torch.randn(5, 1)

        loss = strategy.compute_loss(outputs, labels, context)

        assert loss.item() < 1e-6  # Should be very close to zero


class TestDefaultLossCriterionStrategy:
    """Test suite for DefaultLossCriterionStrategy."""

    def test_with_custom_loss_fn(self):
        """Test that custom loss function is used when provided."""
        custom_loss = nn.MSELoss()
        strategy = DefaultLossCriterionStrategy(loss_fn=custom_loss)

        context = TrainingContext()
        strategy.setup(context)

        assert strategy._criterion is custom_loss

        outputs = torch.randn(10, 1)
        labels = torch.randn(10, 1)

        loss = strategy.compute_loss(outputs, labels, context)
        assert isinstance(loss, torch.Tensor)


class TestLossStrategyComposition:
    """Test combining multiple loss strategies."""

    def test_ce_plus_l2_regularization(self):
        """Test combining cross-entropy with L2 regularization."""
        ce_strategy = CrossEntropyLossStrategy()
        l2_strategy = L2RegularizationStrategy(weight=0.01)

        composite = CompositeLossStrategy([(ce_strategy, 1.0), (l2_strategy, 1.0)])

        context = TrainingContext()
        context.model = nn.Linear(10, 3)
        composite.setup(context)

        outputs = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))

        loss = composite.compute_loss(outputs, labels, context)

        # Should be sum of CE loss and L2 regularization
        ce_loss = ce_strategy.compute_loss(outputs, labels, context)
        l2_loss = l2_strategy.compute_loss(outputs, labels, context)

        expected = ce_loss + l2_loss
        assert torch.allclose(loss, expected, atol=1e-6)
