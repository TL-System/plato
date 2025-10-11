"""
The MNIST dataset from the torchvision package.
"""

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        )

        test_transform = (
            kwargs["test_transform"]
            if "test_transform" in kwargs
            else transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        )
        self.trainset = datasets.MNIST(
            root=_path, train=True, download=True, transform=train_transform
        )
        self.testset = datasets.MNIST(
            root=_path, train=False, download=True, transform=test_transform
        )

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
