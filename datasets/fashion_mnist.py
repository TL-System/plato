
"""
The FashionMNIST dataset.
"""

from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The FashionMNIST dataset."""

    def __init__(self):
        super().__init__()
        self.testset = None

    def read(self, path):
        """Extract the FashionMNIST data using torchvision datasets."""
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)
