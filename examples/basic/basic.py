"""
This example uses a very simple model and the MNIST dataset to show how the model,
the training and validation datasets, as well as the training and testing loops can
be customized in Plato.
"""

from functools import partial

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.clients import simple
from plato.datasources import base
from plato.servers import fedavg
from plato.trainers import basic


class DataSource(base.DataSource):
    """
    A custom datasource with custom training and validation datasets.
    """

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


def main():
    """
    A Plato federated learning training session using a custom model,
    datasource, and trainer.
    """
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
    server.run(client)


if __name__ == "__main__":
    main()
