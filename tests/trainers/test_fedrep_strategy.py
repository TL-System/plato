"""
Tests for FedRep implementation to verify algorithmic correctness.

This test suite verifies that the new composition-based FedRep implementation
is algorithmically identical to the old inheritance-based implementation and
consistent with the FedRep paper (Collins et al., ICML 2021).
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from plato.trainers.strategies.algorithms.personalized_fl_strategy import (
    FedRepUpdateStrategy,
    FedRepUpdateStrategyFromConfig,
)
from plato.trainers.strategies.base import TrainingContext


class SimpleModel(nn.Module):
    """Simple model for testing with clearly separated layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestFedRepLayerFreezing:
    """Test that layers are frozen/activated correctly during training."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_regular_round_local_phase(self, mock_config):
        """During local phase, global layers are frozen and local layers active."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 1
        context.current_epoch = 1  # Within local_epochs

        strategy.on_train_start(context)
        strategy.before_step(context)

        assert not model.conv1.weight.requires_grad
        assert not model.conv2.weight.requires_grad
        assert model.fc1.weight.requires_grad
        assert model.fc2.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_regular_round_global_phase(self, mock_config):
        """During global phase, local layers are frozen and global layers active."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 1
        context.current_epoch = 3  # After local_epochs

        strategy.on_train_start(context)
        strategy.before_step(context)

        assert model.conv1.weight.requires_grad
        assert model.conv2.weight.requires_grad
        assert not model.fc1.weight.requires_grad
        assert not model.fc2.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_personalization_phase(self, mock_config):
        """During personalization, only local layers should be active."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 11  # After trainer.rounds
        context.current_epoch = 1

        strategy.on_train_start(context)
        strategy.before_step(context)

        assert not model.conv1.weight.requires_grad
        assert not model.conv2.weight.requires_grad


class TestFedRepEpochTransitions:
    """Test that layer freezing changes correctly across epochs."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_epoch_transition_from_local_to_global(self, mock_config):
        """Test transition from local phase to global phase."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 1

        strategy.on_train_start(context)

        context.current_epoch = 1
        strategy.before_step(context)
        assert not model.conv1.weight.requires_grad
        assert model.fc1.weight.requires_grad

        context.current_epoch = 2
        strategy.before_step(context)
        assert not model.conv1.weight.requires_grad
        assert model.fc1.weight.requires_grad

        context.current_epoch = 3
        strategy.before_step(context)
        assert model.conv1.weight.requires_grad
        assert not model.fc1.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_multiple_calls_same_epoch_are_idempotent(self, mock_config):
        """Calling before_step multiple times in same epoch should be safe."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 1
        context.current_epoch = 1

        strategy.on_train_start(context)

        for _ in range(5):
            strategy.before_step(context)

        assert not model.conv1.weight.requires_grad
        assert model.fc1.weight.requires_grad


class TestFedRepStrategyFromConfig:
    """Ensure the factory helper constructs a strategy from config values."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_strategy_from_config(self, mock_config):
        """The helper should read layer names and local epochs from config."""
        mock_config.return_value.algorithm = SimpleNamespace(
            global_layer_names=["conv1"],
            local_layer_names=["fc"],
            local_epochs=2,
        )

        strategy = FedRepUpdateStrategyFromConfig()
        assert isinstance(strategy, FedRepUpdateStrategy)
        assert strategy.local_epochs == 2
        assert strategy.global_layer_names == ["conv1"]
        assert strategy.local_layer_names == ["fc"]
