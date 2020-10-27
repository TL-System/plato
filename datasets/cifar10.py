
"""
The CIFAR-10 dataset.
"""

from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The CIFAR-10 dataset."""

    def __init__(self):
        super().__init__()
        self.testset = None


    def read(self, path):
        """Extract the CIFAR-10 data using torchvision datasets."""
        self.trainset = datasets.CIFAR10(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.testset = datasets.CIFAR10(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.labels = list(self.trainset.classes)
