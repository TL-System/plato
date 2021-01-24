"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from datasets import mnist, fashion_mnist, cifar10, cinic10
from config import Config

registered_datasets = {
    'MNIST': mnist,
    'FashionMNIST': fashion_mnist,
    'CIFAR10': cifar10,
    'CINIC10': cinic10
}

if Config().trainer.use_mindspore:
    from datasets import mnist_mindspore
    mindspore_datasets = {'MNIST_mindspore': mnist_mindspore}
    registered_datasets = dict(
        list(registered_datasets.items()) + list(mindspore_datasets.items()))


def get():
    """Get the dataset with the provided name."""
    dataset_name = Config().trainer.dataset
    data_path = Config().trainer.data_path

    logging.info('Dataset: %s', Config().trainer.dataset)

    if dataset_name in registered_datasets:
        dataset = registered_datasets[dataset_name].Dataset(data_path)
    else:
        raise ValueError('No such dataset: {}'.format(dataset_name))

    return dataset
