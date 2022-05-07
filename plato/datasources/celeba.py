"""
The CelebA dataset from the torchvision package.
"""

from torchvision import datasets, transforms

import zipfile
import os
import logging
from plato.config import Config
from plato.datasources import base


class CelebA(datasets.CelebA):

    def _check_integrity(self):
        return True


class DataSource(base.DataSource):
    """The CelebA dataset."""

    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        DataSource.download_celeba(_path)

        _transform = transforms.Compose([transforms.ToTensor()])
        self.trainset = CelebA(root=_path,
                               split='train',
                               target_type=['attr', 'identity'],
                               download=False,
                               transform=_transform)
        self.testset = CelebA(root=_path,
                              split='test',
                              target_type=['attr', 'identity'],
                              download=False,
                              transform=_transform)

    @staticmethod
    def download_celeba(root_path):
        """ Download and unzip all CelebA data points. """
        datapath = os.path.join(root_path, "celeba")
        filename = os.path.join(datapath, "img_align_celeba.zip")
        extracted_path, _ = os.path.splitext(filename)
        if not os.path.exists(extracted_path):
            logging.info("Extracting all images in %s to %s.",
                         "img_align_celeba.zip", extracted_path)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(datapath)
        else:
            logging.info("Path %s already exists.", extracted_path)

    def num_train_examples(self):
        return 162770

    def num_test_examples(self):
        return 19962
