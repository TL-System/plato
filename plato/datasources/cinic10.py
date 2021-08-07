"""
The CINIC-10 dataset.
For more information about CINIC-10, refer to
https://github.com/BayesWatch/cinic-10
"""

import logging
import os

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The CINIC-10 dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        if not os.path.exists(_path):
            logging.info(
                "Downloading the CINIC-10 dataset. This may take a while.")
            url = Config().data.download_url if hasattr(
                Config().data, 'download_url'
            ) else 'https://iqua.ece.toronto.edu/~bli/CINIC-10.tar.gz'
            DataSource.download(url, _path)

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ])
        self.trainset = datasets.ImageFolder(root=os.path.join(_path, 'train'),
                                             transform=_transform)
        self.testset = datasets.ImageFolder(root=os.path.join(_path, 'test'),
                                            transform=_transform)

    def num_train_examples(self):
        return 90000

    def num_test_examples(self):
        return 90000
