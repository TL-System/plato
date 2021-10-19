"""
The CIFAR-10 dataset from the torchvision package.
"""

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The CIFAR-10 dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        _transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.trainset = datasets.CIFAR10(root=_path,
                                         train=True,
                                         download=True,
                                         transform=_transform)
        self.testset = datasets.CIFAR10(root=_path,
                                        train=False,
                                        download=True,
                                        transform=_transform)

    def num_train_examples(self):
        return 50000

    def num_test_examples(self):
        return 10000
