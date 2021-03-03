"""
The MNIST dataset.
"""

import os

from mindspore.dataset.engine.datasets import MnistDataset
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype

from config import Config
from datasources import base


class DataSource(base.DataSource):
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
            DataSource.download(url, self.train_path)

        for url in test_urls:
            DataSource.download(url, self.test_path)

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

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000

    def get_train_set(self):
        dataset = ds.MnistDataset(dataset_dir=self.train_path)
        return DataSource.transform(dataset)

    def get_train_partition(self, num_shards, shard_id):
        """Get a trainset partition for distributed machine learning."""
        dataset = ds.MnistDataset(dataset_dir=self.train_path,
                                  shuffle=False,
                                  num_shards=num_shards,
                                  shard_id=shard_id)

        return DataSource.transform(dataset)

    def get_test_set(self):
        dataset = ds.MnistDataset(dataset_dir=self.test_path)
        return DataSource.transform(dataset)
