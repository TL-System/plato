"""
The CIFAR-10 dataset.
"""

from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The CIFAR-10 dataset."""
    def __init__(self, path):
        super().__init__(path)

        self._transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def num_train_examples():
        return 50000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 10

    def get_train_set(self):
        return datasets.CIFAR10(root=self._path,
                                train=True,
                                download=True,
                                transform=self._transform)

    def get_test_set(self):
        return datasets.CIFAR10(root=self._path,
                                train=False,
                                download=True,
                                transform=self._transform)
