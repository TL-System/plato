"""
The MNIST dataset from the torchvision package.
"""

from pathlib import Path

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

        dataset_root = Path(_path)
        dataset_dir = dataset_root / "MNIST"
        sentinel = dataset_dir / ".download_complete"

        if not sentinel.exists():
            with base.DataSource._download_guard(str(dataset_dir)):
                if not sentinel.exists():
                    datasets.MNIST(
                        root=_path, train=True, download=True, transform=train_transform
                    )
                    datasets.MNIST(
                        root=_path, train=False, download=True, transform=test_transform
                    )
                    sentinel.touch()

        self.trainset = datasets.MNIST(
            root=_path, train=True, download=False, transform=train_transform
        )
        self.testset = datasets.MNIST(
            root=_path, train=False, download=False, transform=test_transform
        )

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
