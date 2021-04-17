import os

os.environ['config_file'] = 'configs/MNIST/fedavg_lenet5.yml'

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from clients import simple
from datasources import base
from servers import fedavg
from trainers import basic


class DataSource(base.DataSource):
    """A custom dataset."""
    def __init__(self):
        super().__init__()

        self.trainset = MNIST("./data",
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST("./data",
                             train=False,
                             download=True,
                             transform=ToTensor())


class Trainer(basic.Trainer):
    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        # the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # the loss
        criterion = nn.CrossEntropyLoss()
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=config['batch_size'],
            sampler=sampler)

        num_epochs = 1
        for __ in range(num_epochs):

            # train
            for train_batch in train_loader:
                x, y = train_batch
                x = x.view(len(x), -1)

                logits = self.model(x)
                loss = criterion(logits, y)
                print("train loss: ", loss.item())

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        # Initializing the data loader for the validation dataset
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False)

        correct = 0
        total = 0

        # testing the model
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                examples = examples.view(len(examples), -1)
                outputs = self.model(examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.cpu()

        accuracy = correct / total
        return accuracy


def main():
    """A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    datasource = DataSource()
    trainer = Trainer(model=model)

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)
    server = fedavg.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
