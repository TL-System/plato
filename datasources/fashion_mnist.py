"""
The FashionMNIST dataset.
"""

from config import Config
from torchvision import datasets, transforms

from datasources import base


class DataSource(base.DataSource):
    """The FashionMNIST dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.trainset = datasets.FashionMNIST(root=_path,
                                              train=True,
                                              download=True,
                                              transform=_transform)

        self.testset = datasets.FashionMNIST(root=_path,
                                             train=False,
                                             download=True,
                                             transform=_transform)

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
