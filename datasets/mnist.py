"""
The MNIST dataset.
"""

from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The MNIST dataset."""

    def __init__(self):
        super().__init__()
        self.testset = None


    @staticmethod
    def num_train_examples():
        return 60000


    @staticmethod
    def num_test_examples():
        return 10000


    @staticmethod
    def num_classes():
        return 10


    def read(self, path):
        """Extract the MNIST data using torchvision datasets."""
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
        ])

        self.trainset = datasets.FashionMNIST(root=path, train=True, 
                            download=True, transform=transform)
        self.testset = datasets.FashionMNIST(root=path, train=False, 
                            transform=transform)
        self.labels = list(self.trainset.classes)
