"""
The MNIST dataset.
"""

import os

from mindspore.dataset import MnistDataset
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""

    def __init__(self):
        super().__init__()
        _path = Config().params["data_path"]

        # Downloading the MNIST dataset from https://ossci-datasets.s3.amazonaws.com/mnist/
        self.train_path = _path + "/MNIST/raw/train"
        self.test_path = _path + "/MNIST/raw/test"

        for data_path in [self.train_path, self.test_path]:
            if not os.path.exists(data_path):
                os.makedirs(data_path)

        train_urls = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        ]

        test_urls = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        ]

        for url in train_urls:
            DataSource.download(url, self.train_path)

        for url in test_urls:
            DataSource.download(url, self.test_path)

    @staticmethod
    def transform(dataset: MnistDataset):
        """Transforming the MNIST dataset."""
        scaled_size = 32
        rescale = 1.0 / 255.0
        shift = 0.0
        rescale_nml = 1 / 0.3081
        shift_nml = -1 * 0.1307 / 0.3081

        # define map operations
        type_cast_op = TypeCast(mstype.int32)
        resize_op = vision.Resize(size=scaled_size, interpolation=Inter.LINEAR)
        rescale_op = vision.Rescale(rescale, shift)
        rescale_nml_op = vision.Rescale(rescale_nml, shift_nml)
        hwc2chw_op = vision.HWC2CHW()

        dataset = dataset.map(operations=type_cast_op, input_columns=["label"])
        dataset = dataset.map(operations=resize_op, input_columns=["image"])
        dataset = dataset.map(operations=rescale_op, input_columns=["image"])
        dataset = dataset.map(operations=rescale_nml_op, input_columns=["image"])
        dataset = dataset.map(operations=hwc2chw_op, input_columns=["image"])

        dataset = dataset.batch(Config().trainer.batch_size, drop_remainder=True)

        return dataset

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000

    def get_train_set(self, sampler):
        dataset = ds.MnistDataset(dataset_dir=self.train_path, sampler=sampler.get())
        return DataSource.transform(dataset)

    def get_test_set(self):
        dataset = ds.MnistDataset(dataset_dir=self.test_path)
        return DataSource.transform(dataset)
