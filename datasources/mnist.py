"""
The MNIST dataset.
"""
from torchvision import datasets, transforms

from datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""
    def __init__(self, path):
        super().__init__(path)

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.trainset = datasets.MNIST(root=self._path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=self._path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
