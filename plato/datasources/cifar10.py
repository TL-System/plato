"""
The CIFAR-10 dataset from the torchvision package.
"""

import logging
import os
import sys

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The CIFAR-10 dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
            )
        )

        test_transform = (
            kwargs["test_transform"]
            if "test_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
            )
        )

        _path = Config().params["data_path"]

        if not os.path.exists(_path):
            if hasattr(Config().server, "do_test") and not Config().server.do_test:
                # If the server is not performing local tests for accuracy, concurrent
                # downloading on the clients may lead to PyTorch errors
                if Config().clients.total_clients > 1:
                    if (
                        not hasattr(Config().data, "concurrent_download")
                        or not Config().data.concurrent_download
                    ):
                        raise ValueError(
                            "The dataset has not yet been downloaded from the Internet. "
                            "Please re-run with '-d' or '--download' first. "
                        )

        self.trainset = datasets.CIFAR10(
            root=_path, train=True, download=True, transform=train_transform
        )
        self.testset = datasets.CIFAR10(
            root=_path, train=False, download=True, transform=test_transform
        )

        if Config().args.download:
            logging.info(
                "The dataset has been successfully downloaded. "
                "Re-run the experiment without '-d' or '--download'."
            )
            sys.exit()

    def num_train_examples(self):
        return 50000

    def num_test_examples(self):
        return 10000
