"""
The FashionMNIST dataset.
"""

from torchvision import datasets, transforms

from datasources import base


class DataSource(base.DataSource):
    """The FashionMNIST dataset."""
    def __init__(self, path):
        super().__init__(path)

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.trainset = datasets.FashionMNIST(root=self._path,
                                              train=True,
                                              download=True,
                                              transform=_transform)

        self.testset = datasets.FashionMNIST(root=self._path,
                                             train=False,
                                             download=True,
                                             transform=_transform)

    @staticmethod
    def num_train_examples():
        return 60000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 10
