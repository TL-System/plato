"""
This example uses a very simple model to show how the model and the server
be customized in Plato and executed in a standalone fashion.

To run this example:

cd examples/customized
uv run custom_server.py -c server.toml
"""

import logging
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.config import Config
from plato.datasources import base
from plato.servers import fedavg
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.trainers.strategies.loss_criterion import CrossEntropyLossStrategy
from plato.trainers.strategies.optimizer import AdamOptimizerStrategy


class CustomServer(fedavg.Server):
    """A custom federated learning server."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        logging.info("A custom server has been initialized.")


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
    """A custom trainer composed with MNIST-specific training and testing strategies."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=CrossEntropyLossStrategy(),
            optimizer_strategy=AdamOptimizerStrategy(lr=1e-3),
            training_step_strategy=MNISTTrainingStepStrategy(),
            testing_strategy=MNISTTestingStrategy(),
        )


def main():
    """A Plato federated learning training session using a custom model."""
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

    server = CustomServer(model=model, datasource=datasource, trainer=trainer)
    server.run()


if __name__ == "__main__":
    main()
