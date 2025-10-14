"""
Strategy-only federated learning examples (no custom server subclasses).

This example demonstrates how to use Plato's server strategies without
requiring any custom server classes or inheritance. All customization is
done through strategy composition.

This is the recommended approach for most use cases as it provides:
- Maximum composability (mix any aggregation with any selection)
- No boilerplate code
- Easy to test and maintain
- Clear separation of concerns
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
    FedAsyncAggregationStrategy,
    FedAvgAggregationStrategy,
    FedNovaAggregationStrategy,
    OortSelectionStrategy,
    RandomSelectionStrategy,
)
from plato.trainers import basic


class DataSource(base.DataSource):
    """A custom datasource with MNIST dataset."""

    def __init__(self):
        super().__init__()
        self.trainset = MNIST("./data", train=True, download=True, transform=ToTensor())
        self.testset = MNIST("./data", train=False, download=True, transform=ToTensor())


class Trainer(basic.Trainer):
    """A basic trainer with simple training loop."""

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

        for examples, labels in train_loader:
            examples = examples.view(len(examples), -1)
            logits = self.model(examples)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

        return correct / total


def get_model_client_server(model=None, datasource=None, trainer=None):
    """Helper function to create model, client, and default server."""
    if model is None:
        model = partial(
            nn.Sequential,
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    if datasource is None:
        datasource = DataSource
    if trainer is None:
        trainer = Trainer

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)
    server = fedavg.Server(model=model, datasource=datasource, trainer=trainer)

    return model, client, server


def example_1_default():
    """Example 1: Default strategies (FedAvg + Random)."""
    print("\n" + "=" * 70)
    print("Example 1: Default Strategies")
    print("=" * 70)
    print("Using: FedAvg aggregation + Random client selection")

    model, client, server = get_model_client_server()

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print("\n✓ Server ready to run")


def example_2_fednova():
    """Example 2: FedNova aggregation with random selection."""
    print("\n" + "=" * 70)
    print("Example 2: FedNova Aggregation")
    print("=" * 70)
    print("Using: FedNova aggregation + Random client selection")

    model = partial(
        nn.Sequential, nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )
    client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)

    server = fedavg.Server(
        model=model,
        datasource=DataSource,
        trainer=Trainer,
        aggregation_strategy=FedNovaAggregationStrategy(),
    )

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print("\n✓ Server ready to run")


def example_3_oort():
    """Example 3: FedAvg aggregation with Oort selection."""
    print("\n" + "=" * 70)
    print("Example 3: Oort Client Selection")
    print("=" * 70)
    print("Using: FedAvg aggregation + Oort client selection")

    model = partial(
        nn.Sequential, nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )
    client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)

    server = fedavg.Server(
        model=model,
        datasource=DataSource,
        trainer=Trainer,
        client_selection_strategy=OortSelectionStrategy(
            exploration_factor=0.3,
            desired_duration=100.0,
            blacklist_num=10,
        ),
    )

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print(
        f"  Exploration factor: {server.client_selection_strategy.exploration_factor}"
    )
    print("\n✓ Server ready to run")


def example_4_fedasync():
    """Example 4: FedAsync aggregation with random selection."""
    print("\n" + "=" * 70)
    print("Example 4: FedAsync Aggregation")
    print("=" * 70)
    print("Using: FedAsync aggregation + Random client selection")

    model = partial(
        nn.Sequential, nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )
    client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)

    server = fedavg.Server(
        model=model,
        datasource=DataSource,
        trainer=Trainer,
        aggregation_strategy=FedAsyncAggregationStrategy(
            mixing_hyperparameter=0.9,
            adaptive_mixing=True,
            staleness_func_type="polynomial",
            staleness_func_params={"a": 0.5},
        ),
    )

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print(f"  Mixing hyperparameter: {server.aggregation_strategy.mixing_hyperparam}")
    print(f"  Adaptive mixing: {server.aggregation_strategy.adaptive_mixing}")
    print("\n✓ Server ready to run")


def example_5_afl():
    """Example 5: FedAvg aggregation with AFL selection."""
    print("\n" + "=" * 70)
    print("Example 5: Active Federated Learning (AFL)")
    print("=" * 70)
    print("Using: FedAvg aggregation + AFL client selection")

    model = partial(
        nn.Sequential, nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )
    client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)

    server = fedavg.Server(
        model=model,
        datasource=DataSource,
        trainer=Trainer,
        client_selection_strategy=AFLSelectionStrategy(
            alpha1=0.75,
            alpha2=0.01,
            alpha3=0.1,
        ),
    )

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print(
        f"  Alpha parameters: α1={server.client_selection_strategy.alpha1}, "
        f"α2={server.client_selection_strategy.alpha2}, α3={server.client_selection_strategy.alpha3}"
    )
    print("\n✓ Server ready to run")


def example_6_combined():
    """Example 6: FedNova aggregation with Oort selection."""
    print("\n" + "=" * 70)
    print("Example 6: Combined Strategies (FedNova + Oort)")
    print("=" * 70)
    print("Using: FedNova aggregation + Oort client selection")

    model = partial(
        nn.Sequential, nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    )
    client = simple.Client(model=model, datasource=DataSource, trainer=Trainer)

    server = fedavg.Server(
        model=model,
        datasource=DataSource,
        trainer=Trainer,
        aggregation_strategy=FedNovaAggregationStrategy(),
        client_selection_strategy=OortSelectionStrategy(
            exploration_factor=0.3,
            desired_duration=100.0,
        ),
    )

    print(f"\nServer configuration:")
    print(f"  Aggregation: {type(server.aggregation_strategy).__name__}")
    print(f"  Selection: {type(server.client_selection_strategy).__name__}")
    print(
        f"  Exploration factor: {server.client_selection_strategy.exploration_factor}"
    )
    print("\n✓ Server ready to run")
    print("\n💡 This combination was impossible with inheritance-based approach!")


def main():
    """
    Demonstrates all strategy-only examples.

    To actually run training with one of these configurations:
    1. Uncomment the server.run(client) line in the desired example
    2. Run: python strategies_only.py -c config.yml
    """
    print("\n" + "=" * 70)
    print("STRATEGY-ONLY FEDERATED LEARNING EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating pure composition approach (no inheritance):")

    example_1_default()
    example_2_fednova()
    example_3_oort()
    example_4_fedasync()
    example_5_afl()
    example_6_combined()

    print("\n" + "=" * 70)
    print("KEY BENEFITS OF STRATEGY-ONLY APPROACH")
    print("=" * 70)
    print("✓ No custom server classes needed")
    print("✓ Mix any aggregation with any selection")
    print("✓ No code duplication")
    print("✓ Easy to test strategies independently")
    print("✓ Clear and maintainable")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("TO RUN TRAINING")
    print("=" * 70)
    print("1. Create a config.yml file (see examples/basic/config.yml)")
    print("2. Uncomment server.run(client) in the desired example")
    print("3. Run: python strategies_only.py -c config.yml")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
