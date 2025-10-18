"""
This example demonstrates the strategy-based server API in Plato.

It shows how to use different aggregation and client selection strategies
by composing them with the server.
"""

import sys
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import example client-selection strategies for demonstration purposes.
try:
    from examples.client_selection.afl.afl_selection_strategy import (
        AFLSelectionStrategy,
    )
    from examples.client_selection.oort.oort_selection_strategy import (
        OortSelectionStrategy,
    )
except ModuleNotFoundError:
    EXAMPLES_DIR = Path(__file__).resolve().parents[1]
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from client_selection.afl.afl_selection_strategy import AFLSelectionStrategy
    from client_selection.oort.oort_selection_strategy import OortSelectionStrategy
from plato.clients import simple
from plato.config import Config
from plato.datasources import base
from plato.servers import fedavg
from plato.servers.strategies import (
    FedAvgAggregationStrategy,
    FedNovaAggregationStrategy,
    RandomSelectionStrategy,
)
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.trainers.strategies.loss_criterion import CrossEntropyLossStrategy
from plato.trainers.strategies.optimizer import AdamOptimizerStrategy


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        Config()
        base_path = Path(Config.params.get("base_path", "./runtime"))
        data_dir = Path(Config.params.get("data_path", base_path / "data"))
        self.trainset = MNIST(
            str(data_dir), train=True, download=True, transform=ToTensor()
        )
        self.testset = MNIST(
            str(data_dir), train=False, download=True, transform=ToTensor()
        )


class MNISTTrainingStepStrategy(TrainingStepStrategy):
    """Custom training step that flattens MNIST images and prints the loss."""

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform a single MNIST training step."""
        optimizer.zero_grad()

        flattened_examples = examples.view(examples.size(0), -1)
        outputs = model(flattened_examples)
        loss = loss_criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f"train loss: {loss.item():.6f}")

        return loss


class MNISTTestingStrategy(TestingStrategy):
    """Testing strategy that flattens MNIST images before evaluation."""

    def test_model(self, model, config, testset, sampler, context):
        """Evaluate the model with flattened MNIST images."""
        batch_size = config.get("batch_size", 32)

        if sampler is not None and hasattr(sampler, "get") and callable(sampler.get):
            sampler = sampler.get()

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        model.to(context.device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                flattened_examples = examples.view(examples.size(0), -1)
                outputs = model(flattened_examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy


class Trainer(ComposableTrainer):
    """
    A custom trainer composed with MNIST-specific training and testing strategies.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=CrossEntropyLossStrategy(),
            optimizer_strategy=AdamOptimizerStrategy(lr=1e-3),
            training_step_strategy=MNISTTrainingStepStrategy(),
            testing_strategy=MNISTTestingStrategy(),
        )


def example_1_default_strategies():
    """
    Example 1: Server with default strategies (FedAvg + Random).
    This is the same as not specifying strategies at all.
    """
    print("\n" + "=" * 60)
    print("Example 1: Default Strategies (FedAvg + Random)")
    print("=" * 60)

    model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource
    trainer = Trainer

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)
    server = fedavg.Server(model=model, datasource=datasource, trainer=trainer)

    print(f"Server created with:")
    print(f"  - Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  - Selection: {type(server.client_selection_strategy).__name__}")
    print("Ready to run with: server.run(client)")


def example_2_custom_aggregation():
    """
    Example 2: Server with custom aggregation strategy (FedNova).
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Aggregation (FedNova + Random)")
    print("=" * 60)

    model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource
    trainer = Trainer

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)

    # Use FedNova aggregation strategy
    server = fedavg.Server(
        model=model,
        datasource=datasource,
        trainer=trainer,
        aggregation_strategy=FedNovaAggregationStrategy(),
    )

    print(f"Server created with:")
    print(f"  - Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  - Selection: {type(server.client_selection_strategy).__name__}")
    print("Ready to run with: server.run(client)")


def example_3_custom_selection():
    """
    Example 3: Server with custom client selection strategy (Oort).
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Selection (FedAvg + Oort)")
    print("=" * 60)

    model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource
    trainer = Trainer

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)

    # Use Oort client selection strategy
    server = fedavg.Server(
        model=model,
        datasource=datasource,
        trainer=trainer,
        client_selection_strategy=OortSelectionStrategy(
            exploration_factor=0.3, desired_duration=100.0, blacklist_num=10
        ),
    )

    print(f"Server created with:")
    print(f"  - Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  - Selection: {type(server.client_selection_strategy).__name__}")
    print(
        f"  - Oort params: exploration={server.client_selection_strategy.exploration_factor}"
    )
    print("Ready to run with: server.run(client)")


def example_4_both_custom():
    """
    Example 4: Server with both custom strategies (FedNova + AFL).
    """
    print("\n" + "=" * 60)
    print("Example 4: Both Custom (FedNova + AFL)")
    print("=" * 60)

    model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource
    trainer = Trainer

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)

    # Use both custom strategies
    server = fedavg.Server(
        model=model,
        datasource=datasource,
        trainer=trainer,
        aggregation_strategy=FedNovaAggregationStrategy(),
        client_selection_strategy=AFLSelectionStrategy(
            alpha1=0.75, alpha2=0.01, alpha3=0.1
        ),
    )

    print(f"Server created with:")
    print(f"  - Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  - Selection: {type(server.client_selection_strategy).__name__}")
    print(
        f"  - AFL params: alpha1={server.client_selection_strategy.alpha1}, "
        f"alpha2={server.client_selection_strategy.alpha2}"
    )
    print("Ready to run with: server.run(client)")


def main():
    """
    Demonstrates different ways to use server strategies.
    """
    print("\nThis example shows how to compose different strategies with servers.")
    print("Strategies allow mixing and matching aggregation and client selection")
    print("algorithms without requiring inheritance or code duplication.")

    example_1_default_strategies()
    example_2_custom_aggregation()
    example_3_custom_selection()
    example_4_both_custom()


if __name__ == "__main__":
    main()
