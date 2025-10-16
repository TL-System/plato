"""Tests for the split learning testing strategy."""

from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from plato.trainers.split_learning import SplitLearningTestingStrategy
from plato.trainers.strategies.base import TrainingContext


class DummySampler:
    """Sampler that exposes a `get` method returning a PyTorch sampler."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get(self):
        return torch.utils.data.SequentialSampler(self.dataset)


def test_split_learning_testing_strategy_accepts_custom_sampler():
    """Ensure the testing strategy consumes samplers exposing `get`."""
    num_samples, num_features, num_classes = 6, 4, 3
    features = torch.randn(num_samples, num_features)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(features, labels)
    sampler = DummySampler(dataset)

    strategy = SplitLearningTestingStrategy()
    model = torch.nn.Linear(num_features, num_classes)

    context = TrainingContext()
    context.device = torch.device("cpu")
    context.state["trainer"] = SimpleNamespace()

    config = {"batch_size": 2}

    accuracy = strategy.test_model(model, config, dataset, sampler, context)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
