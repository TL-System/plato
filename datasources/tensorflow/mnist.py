"""
The MNIST dataset.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets

from config import Config
from datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        (x_train, y_train), (x_test,
                             y_test) = datasets.mnist.load_data(path=_path)

        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))

        # Prepare the training dataset.
        self.trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.trainset = self.trainset.shuffle(buffer_size=1024).batch(
            Config().trainer.batch_size)

        # Prepare the test dataset.
        self.testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.testset = self.testset.batch(Config().trainer.batch_size)
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
