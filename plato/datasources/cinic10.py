"""
The CINIC-10 dataset.

For more information about CINIC-10, refer to:

https://github.com/BayesWatch/cinic-10
"""

import logging
import os

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import _image_folder, base


class DataSource(base.DataSource):
    """The CINIC-10 dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        def ready():
            return _image_folder.prepared_ready(_path)

        if not ready():
            logging.info("Downloading the CINIC-10 dataset. This may take a while.")
            url = (
                Config().data.download_url
                if hasattr(Config().data, "download_url")
                else "http://iqua.ece.toronto.edu/baochun/CINIC-10.tar.gz"
            )
            DataSource.download(url, _path, ready=ready)

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.47889522, 0.47227842, 0.43047404],
                            [0.24205776, 0.23828046, 0.25874835],
                        ),
                    ]
                )
            )
        )
        test_transform = kwargs.get("test_transform", train_transform)

        self.trainset = datasets.ImageFolder(
            root=os.path.join(_path, "train"), transform=train_transform
        )
        self.testset = _image_folder.evaluation_folder(
            os.path.join(_path, "test"), self.trainset, transform=test_transform
        )
