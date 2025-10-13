"""
Tests for FedRep implementation to verify algorithmic correctness.

This test suite verifies that the new composition-based FedRep implementation
is algorithmically identical to the old inheritance-based implementation and
consistent with the FedRep paper (Collins et al., ICML 2021).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from plato.config import Config
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
        """Test that during local phase, global layers are frozen and local layers are active."""
        # Setup mock config
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

        # Initialize
        strategy.on_train_start(context)
        strategy.before_step(context)

        # Check that global layers are frozen
        assert not model.conv1.weight.requires_grad
        assert not model.conv2.weight.requires_grad

        # Check that local layers are active
        assert model.fc1.weight.requires_grad
        assert model.fc2.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_regular_round_global_phase(self, mock_config):
        """Test that during global phase, local layers are frozen and global layers are active."""
        # Setup mock config
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

        # Initialize
        strategy.on_train_start(context)
        strategy.before_step(context)

        # Check that global layers are active
        assert model.conv1.weight.requires_grad
        assert model.conv2.weight.requires_grad

        # Check that local layers are frozen
        assert not model.fc1.weight.requires_grad
        assert not model.fc2.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_personalization_phase(self, mock_config):
        """Test that during personalization, only local layers are active."""
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

        # Initialize - should freeze global layers
        strategy.on_train_start(context)

        # Check that global layers are frozen
        assert not model.conv1.weight.requires_grad
        assert not model.conv2.weight.requires_grad

        # Check that local layers remain active (not explicitly activated, but not frozen)
        # Note: before_step should NOT be called during personalization
        strategy.before_step(context)

        # Global layers should still be frozen
        assert not model.conv1.weight.requires_grad
        assert not model.conv2.weight.requires_grad


class TestFedRepEpochTransitions:
    """Test that layer freezing changes correctly across epochs."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_epoch_transition_from_local_to_global(self, mock_config):
        """Test transition from local phase to global phase."""
        # Setup mock config
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

        # Epoch 1 - local phase
        context.current_epoch = 1
        strategy.before_step(context)
        assert not model.conv1.weight.requires_grad  # Global frozen
        assert model.fc1.weight.requires_grad  # Local active

        # Epoch 2 - still local phase
        context.current_epoch = 2
        strategy.before_step(context)
        assert not model.conv1.weight.requires_grad  # Global frozen
        assert model.fc1.weight.requires_grad  # Local active

        # Epoch 3 - transition to global phase
        context.current_epoch = 3
        strategy.before_step(context)
        assert model.conv1.weight.requires_grad  # Global active
        assert not model.fc1.weight.requires_grad  # Local frozen

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_multiple_calls_same_epoch_are_idempotent(self, mock_config):
        """Test that calling before_step multiple times in same epoch is safe."""
        # Setup mock config
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

        # Call multiple times (simulating multiple batches in same epoch)
        for _ in range(10):
            strategy.before_step(context)

        # Should still have correct state
        assert not model.conv1.weight.requires_grad  # Global frozen
        assert model.fc1.weight.requires_grad  # Local active


class TestFedRepPersonalizationEpochs:
    """Test that personalization epochs configuration is applied correctly."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_personalization_epochs_applied(self, mock_config):
        """Test that personalization epochs override regular epochs."""
        # Setup mock config
        mock_trainer = MagicMock()
        mock_trainer.rounds = 10
        mock_algorithm = MagicMock()
        mock_algorithm.personalization.epochs = 5
        mock_config.return_value.trainer = mock_trainer
        mock_config.return_value.algorithm = mock_algorithm

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 11  # Personalization phase
        context.config = {"epochs": 3}  # Original epochs

        # Initialize
        strategy.on_train_start(context)

        # Check that config was modified
        assert context.config["epochs"] == 5

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_personalization_epochs_not_applied_during_regular_rounds(
        self, mock_config
    ):
        """Test that personalization epochs are not applied during regular rounds."""
        mock_trainer = MagicMock()
        mock_trainer.rounds = 10
        mock_algorithm = MagicMock()
        mock_algorithm.personalization.epochs = 5
        mock_config.return_value.trainer = mock_trainer
        mock_config.return_value.algorithm = mock_algorithm

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 5  # Regular round
        context.config = {"epochs": 3}

        # Initialize
        strategy.on_train_start(context)

        # Check that config was NOT modified
        assert context.config["epochs"] == 3


class TestFedRepCleanup:
    """Test that model state is properly cleaned up after training."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_all_layers_reactivated_after_training(self, mock_config):
        """Test that all layers are reactivated after training ends."""
        # Setup mock config
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

        # Train and freeze some layers
        strategy.on_train_start(context)
        strategy.before_step(context)

        # Some layers should be frozen
        assert not model.conv1.weight.requires_grad

        # End training
        strategy.on_train_end(context)

        # All layers should be reactivated
        assert model.conv1.weight.requires_grad
        assert model.conv2.weight.requires_grad
        assert model.fc1.weight.requires_grad
        assert model.fc2.weight.requires_grad


class TestFedRepFromConfig:
    """Test FedRepUpdateStrategyFromConfig initialization."""

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_from_config_initialization(self, mock_config):
        """Test that strategy can be initialized from config."""
        mock_algorithm = MagicMock()
        mock_algorithm.global_layer_names = ["conv1", "conv2"]
        mock_algorithm.local_layer_names = ["fc1", "fc2"]
        mock_algorithm.local_epochs = 3
        mock_config.return_value.algorithm = mock_algorithm

        strategy = FedRepUpdateStrategyFromConfig()

        assert strategy.global_layer_names == ["conv1", "conv2"]
        assert strategy.local_layer_names == ["fc1", "fc2"]
        assert strategy.local_epochs == 3

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_from_config_default_local_epochs(self, mock_config):
        """Test that local_epochs defaults to 1 if not in config."""
        mock_algorithm = MagicMock()
        mock_algorithm.global_layer_names = ["conv1", "conv2"]
        mock_algorithm.local_layer_names = ["fc1", "fc2"]
        # No local_epochs attribute
        del mock_algorithm.local_epochs
        mock_config.return_value.algorithm = mock_algorithm

        strategy = FedRepUpdateStrategyFromConfig()

        assert strategy.local_epochs == 1

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_from_config_missing_global_layer_names(self, mock_config):
        """Test that error is raised if global_layer_names is missing."""
        mock_algorithm = MagicMock()
        del mock_algorithm.global_layer_names
        mock_algorithm.local_layer_names = ["fc1", "fc2"]
        mock_config.return_value.algorithm = mock_algorithm

        with pytest.raises(ValueError, match="global_layer_names is required"):
            FedRepUpdateStrategyFromConfig()

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_from_config_missing_local_layer_names(self, mock_config):
        """Test that error is raised if local_layer_names is missing."""
        mock_algorithm = MagicMock()
        mock_algorithm.global_layer_names = ["conv1", "conv2"]
        del mock_algorithm.local_layer_names
        mock_config.return_value.algorithm = mock_algorithm

        with pytest.raises(ValueError, match="local_layer_names is required"):
            FedRepUpdateStrategyFromConfig()


class TestFedRepAlgorithmicEquivalence:
    """
    Tests to verify algorithmic equivalence with the original implementation.

    These tests simulate the training flow and verify that layer states match
    what the old inheritance-based implementation would produce.
    """

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_complete_training_round_simulation(self, mock_config):
        """Simulate a complete training round with 5 epochs."""
        # Setup mock config
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

        # Start training
        strategy.on_train_start(context)

        # Track layer states through epochs
        layer_states = []

        for epoch in range(1, 6):
            context.current_epoch = epoch

            # Simulate multiple batches per epoch (10 batches)
            for batch in range(10):
                strategy.before_step(context)

                # Record state only on first batch of each epoch
                if batch == 0:
                    layer_states.append(
                        {
                            "epoch": epoch,
                            "global_frozen": not model.conv1.weight.requires_grad,
                            "local_frozen": not model.fc1.weight.requires_grad,
                        }
                    )

        # End training
        strategy.on_train_end(context)

        # Verify expected states
        # Epochs 1-2: local training (global frozen, local active)
        assert layer_states[0]["global_frozen"] is True
        assert layer_states[0]["local_frozen"] is False
        assert layer_states[1]["global_frozen"] is True
        assert layer_states[1]["local_frozen"] is False

        # Epochs 3-5: global training (global active, local frozen)
        assert layer_states[2]["global_frozen"] is False
        assert layer_states[2]["local_frozen"] is True
        assert layer_states[3]["global_frozen"] is False
        assert layer_states[3]["local_frozen"] is True
        assert layer_states[4]["global_frozen"] is False
        assert layer_states[4]["local_frozen"] is True

        # After training: all active
        assert model.conv1.weight.requires_grad
        assert model.fc1.weight.requires_grad

    @patch("plato.trainers.strategies.algorithms.personalized_fl_strategy.Config")
    def test_complete_personalization_round_simulation(self, mock_config):
        """Simulate a complete personalization round."""
        mock_config.return_value.trainer.rounds = 10

        model = SimpleModel()
        strategy = FedRepUpdateStrategy(
            global_layer_names=["conv1", "conv2"],
            local_layer_names=["fc1", "fc2"],
            local_epochs=2,
        )

        context = TrainingContext()
        context.model = model
        context.current_round = 11  # Personalization

        # Start training
        strategy.on_train_start(context)

        # Global layers should be frozen throughout
        assert not model.conv1.weight.requires_grad

        # Simulate epochs
        for epoch in range(1, 4):
            context.current_epoch = epoch
            for _ in range(10):  # Multiple batches
                strategy.before_step(context)

                # Global should remain frozen
                assert not model.conv1.weight.requires_grad

        # End training
        strategy.on_train_end(context)

        # All should be reactivated
        assert model.conv1.weight.requires_grad
        assert model.fc1.weight.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
