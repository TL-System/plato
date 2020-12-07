"""
The MNIST dataset.
"""
import logging
from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The MNIST dataset."""
    def __init__(self, path):
        super().__init__(path)

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    @staticmethod
    def num_train_examples():
        return 60000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 10

    def get_train_set(self):
        return datasets.MNIST(root=self._path,
                              train=True,
                              download=True,
                              transform=self._transform)

    def get_test_set(self):
        return datasets.MNIST(root=self._path,
                              train=False,
                              download=True,
                              transform=self._transform)
