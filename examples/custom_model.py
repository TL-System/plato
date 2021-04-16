import os

os.environ['config_file'] = 'configs/MNIST/fedavg_lenet5.yml'

from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from clients import simple
from datasources import base
from servers import fedavg


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

    client = simple.Client(model=model, datasource=datasource)
    server = fedavg.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
