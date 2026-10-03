"""Tests for the split learning testing strategy."""

from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from plato.trainers.split_learning import SplitLearningTestingStrategy
from plato.trainers.strategies.base import TrainingContext
from tests.integration.utils import build_minimal_config, configure_environment


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


def test_fresh_public_split_evaluation_and_empty_partition(tmp_path):
    """Evaluation does not depend on installing a training-only callback."""
    from plato.trainers.split_learning import Trainer

    config = build_minimal_config(trainer_type="split_learning")
    config["trainer"]["batch_size"] = 2
    with configure_environment(config, runtime_root=tmp_path):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        trainer = Trainer(model=model)
        trainer.device = trainer.context.device = torch.device("cpu")
        dataset = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        assert trainer.test(dataset) == 1.0
        empty = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        assert trainer.test(empty) == 0.0
