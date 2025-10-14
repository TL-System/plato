"""
This example demonstrates the strategy-based server API in Plato.

It shows how to use different aggregation and client selection strategies
by composing them with the server, rather than using inheritance.
"""

from functools import partial

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.clients import simple
from plato.datasources import base
from plato.servers import fedavg
from plato.servers.strategies import (
    AFLSelectionStrategy,
    FedAvgAggregationStrategy,
    FedNovaAggregationStrategy,
    OortSelectionStrategy,
    RandomSelectionStrategy,
)
from plato.trainers import basic


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        self.trainset = MNIST("./data", train=True, download=True, transform=ToTensor())
        self.testset = MNIST("./data", train=False, download=True, transform=ToTensor())


class Trainer(basic.Trainer):
    """A custom trainer with custom training and testing loops."""

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """A custom training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=config["batch_size"],
            sampler=sampler,
        )

        num_epochs = 1
        for __ in range(num_epochs):
            for examples, labels in train_loader:
                examples = examples.view(len(examples), -1)

                logits = self.model(examples)
                loss = criterion(logits, labels)
                print("train loss: ", loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """A custom testing loop."""
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config["batch_size"], shuffle=False
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                examples = examples.view(len(examples), -1)
                outputs = self.model(examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy


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

    To actually run one of these examples with training:
    - Uncomment one of the example functions below
    - Run with: python basic_with_strategies.py -c config.yml
    """
    print("\n" + "=" * 70)
    print("PLATO SERVER STRATEGIES DEMONSTRATION")
    print("=" * 70)
    print("\nThis example shows how to compose different strategies with servers.")
    print("Strategies allow mixing and matching aggregation and client selection")
    print("algorithms without requiring inheritance or code duplication.")

    # Show all examples
    example_1_default_strategies()
    example_2_custom_aggregation()
    example_3_custom_selection()
    example_4_both_custom()

    print("\n" + "=" * 70)
    print("To run one of these examples:")
    print("1. Edit this file to uncomment one server.run(client) call")
    print("2. Run: python basic_with_strategies.py -c config.yml")
    print("=" * 70 + "\n")

    # Uncomment ONE of the following to actually run training:

    # Option 1: Default strategies
    # model = partial(nn.Sequential, nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    # client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)
    # server = fedavg.Server(model=model, datasource=DataSource, trainer=Trainer)
    # server.run(client)

    # Option 2: FedNova aggregation
    # model = partial(nn.Sequential, nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    # client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)
    # server = fedavg.Server(
    #     model=model, datasource=DataSource, trainer=Trainer,
    #     aggregation_strategy=FedNovaAggregationStrategy()
    # )
    # server.run(client)

    # Option 3: Oort client selection
    # model = partial(nn.Sequential, nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    # client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)
    # server = fedavg.Server(
    #     model=model, datasource=DataSource, trainer=Trainer,
    #     client_selection_strategy=OortSelectionStrategy(exploration_factor=0.3)
    # )
    # server.run(client)


if __name__ == "__main__":
    main()
