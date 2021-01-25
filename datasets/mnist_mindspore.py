"""
The MNIST dataset.
"""

import os
import sys
import logging
import gzip
from urllib.parse import urlparse
from mindspore.dataset.engine.datasets import MnistDataset
import requests

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype

from config import Config
from datasets import base


class Dataset(base.Dataset):
    """The MNIST dataset."""
    def __init__(self, path):
        super().__init__(path)

        # Downloading the MNIST dataset from http://yann.lecun.com/exdb/mnist/
        self.train_path = self._path + "/MNIST/raw/train/"
        self.test_path = self._path + "/MNIST/raw/test/"

        for data_path in [self.train_path, self.test_path]:
            if not os.path.exists(data_path):
                os.makedirs(data_path)

        train_urls = [
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        ]

        test_urls = [
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        ]

        for url in train_urls:
            Dataset.download(url, self.train_path)

        for url in test_urls:
            Dataset.download(url, self.test_path)

    @staticmethod
    def download(url, data_path):
        """downloading the MNIST dataset."""
        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split('/')[-1])

        if not os.path.exists(file_name.replace('.gz', '')):
            logging.info("Downloading %s.", url)

            res = requests.get(url, stream=True, verify=False)
            total_size = int(res.headers["Content-Length"])
            downloaded_size = 0

            with open(file_name, "wb+") as file:
                for chunk in res.iter_content(chunk_size=1024):
                    downloaded_size += len(chunk)
                    file.write(chunk)
                    file.flush()
                    done = int(100 * downloaded_size / total_size)
                    # show download progress
                    sys.stdout.write("\r[{}{}] {:.2f}%".format(
                        "█" * done, " " * (100 - done),
                        100 * downloaded_size / total_size))
                    sys.stdout.flush()
                sys.stdout.write("\n")

            # Unzip the compressed file just downloaded
            unzipped_file = open(file_name.replace('.gz', ''), 'wb')
            zipped_file = gzip.GzipFile(file_name)
            unzipped_file.write(zipped_file.read())
            zipped_file.close()

            os.remove(file_name)

    @staticmethod
    def transform(dataset: MnistDataset):
        """Transforming the MNIST dataset."""
        resize_height, resize_width = 32, 32
        rescale = 1.0 / 255.0
        shift = 0.0
        rescale_nml = 1 / 0.3081
        shift_nml = -1 * 0.1307 / 0.3081

        resize_op = CV.Resize((resize_height, resize_width),
                              interpolation=Inter.LINEAR)
        rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
        rescale_op = CV.Rescale(rescale, shift)
        hwc2chw_op = CV.HWC2CHW()
        type_cast_op = C.TypeCast(mstype.int32)

        dataset = dataset.map(operations=type_cast_op, input_columns="label")
        dataset = dataset.map(operations=resize_op, input_columns="image")
        dataset = dataset.map(operations=rescale_op, input_columns="image")
        dataset = dataset.map(operations=rescale_nml_op, input_columns="image")
        dataset = dataset.map(operations=hwc2chw_op, input_columns="image")

        dataset = dataset.batch(Config().trainer.batch_size,
                                drop_remainder=True)

        return dataset

    @staticmethod
    def num_train_examples():
        return 60000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_classes():
        return 10

    def get_train_set(self):
        dataset = ds.MnistDataset(dataset_dir=self.train_path)
        return Dataset.transform(dataset)

    def get_train_partition(self, num_shards, shard_id):
        """Get a trainset partition for distributed machine learning."""
        dataset = ds.MnistDataset(dataset_dir=self.train_path,
                                  shuffle=False,
                                  num_shards=num_shards,
                                  shard_id=shard_id)

        return Dataset.transform(dataset)

    def get_test_set(self):
        dataset = ds.MnistDataset(dataset_dir=self.test_path)
        return Dataset.transform(dataset)
