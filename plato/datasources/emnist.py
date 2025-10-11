"""
The Extended MNIST (EMNIST) dataset from the torchvision package.
"""

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The EMNIST dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(
                            degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                )
            )
        )

        test_transform = (
            kwargs["test_transform"]
            if "test_transform" in kwargs
            else (
                transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
                )
            )
        )

        self.trainset = datasets.EMNIST(
            root=_path,
            split="balanced",
            train=True,
            download=True,
            transform=train_transform,
        )
        self.testset = datasets.EMNIST(
            root=_path,
            split="balanced",
            train=False,
            download=True,
            transform=test_transform,
        )

    def num_train_examples(self):
        return 112800

    def num_test_examples(self):
        return 18800
