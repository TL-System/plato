"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging
from collections import OrderedDict
from config import Config

from datasources import (
    mnist,
    fashion_mnist,
    cifar10,
    cinic10,
    huggingface,
)

if hasattr(Config().trainer, 'use_mindspore'):
    from datasources.mindspore import (
        mnist as mnist_mindspore, )

    registered_datasources = OrderedDict([
        ('MNIST', mnist_mindspore),
    ])
else:
    registered_datasources = OrderedDict([
        ('MNIST', mnist),
        ('FashionMNIST', fashion_mnist),
        ('CIFAR10', cifar10),
        ('CINIC10', cinic10),
        ('HuggingFace', huggingface),
    ])

    if Config().data.datasource == 'COCO':
        from datasources import coco
        coco_datasources = {'COCO': coco}
        registered_datasources = OrderedDict(
            list(registered_datasources.items()) +
            list(coco_datasources.items()))


def get():
    """Get the data source with the provided name."""
    datasource_name = Config().data.datasource

    logging.info("Data source: %s", Config().data.datasource)

    if datasource_name in registered_datasources:
        dataset = registered_datasources[datasource_name].DataSource()
    else:
        raise ValueError('No such data source: {}'.format(datasource_name))

    return dataset
