"""
The Tiny ImageNet 200 Classification dataset.

Tiny ImageNet contains 100000 images of 200 classes (500 for each class)
downsized to 64×64 colored images.
Each class has 500 training images, 50 validation images and 50 test images.
"""

import logging
import os

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The Tiny ImageNet 200 dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        if not os.path.exists(_path):
            logging.info(
                "Downloading the Tiny ImageNet 200 dataset. This may take a while."
            )
            url = Config().data.download_url if hasattr(
                Config().data, 'download_url'
            ) else 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            DataSource.download(url, _path)

        _transform = transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.trainset = datasets.ImageFolder(root=os.path.join(_path, 'train'),
                                             transform=_transform)
        self.testset = datasets.ImageFolder(root=os.path.join(_path, 'test'),
                                            transform=_transform)

    def num_train_examples(self):
        return 100000

    def num_test_examples(self):
        return 10000
