"""
The MNIST dataset.
"""
import tensorflow as tf
import tensorflow_datasets as tfds

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""
    def __init__(self):
        super().__init__()

        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        ds_train = ds_train.map(
            DataSource.normalize_img,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(Config().trainer.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.map(DataSource.normalize_img,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(Config().trainer.batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        self.trainset = ds_train
        self.testset = ds_test

    @staticmethod
    def normalize_img(image, label):
        """ Normalizes images: uint8 -> float32. """
        return tf.cast(image, tf.float32) / 255., label

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000

    @staticmethod
    def input_shape():
        """ Returns the input shape of the MNIST dataset, useful for building
        a TF model. """
        return [60000, 28, 28, 1]
