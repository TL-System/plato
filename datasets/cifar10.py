
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


    @staticmethod
    def num_train_examples():
        return 50000


    @staticmethod
    def num_test_examples():
        return 10000


    @staticmethod
    def num_classes():
        return 10


    def read(self, path):
        """Extract the CIFAR-10 data using torchvision datasets."""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.trainset = datasets.CIFAR10(root=path, train=True,
                            download=True, transform=transform)
        self.testset = datasets.CIFAR10(root=path, train=False,
                            transform=transform)

        self.labels = list(self.trainset.classes)
