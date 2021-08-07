"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging
from collections import OrderedDict

from plato.config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from plato.datasources.mindspore import (
        mnist as mnist_mindspore, )

    registered_datasources = OrderedDict([
        ('MNIST', mnist_mindspore),
    ])
elif hasattr(Config().trainer, 'use_tensorflow'):
    from plato.datasources.tensorflow import (
        mnist as mnist_tensorflow,
        fashion_mnist as fashion_mnist_tensorflow,
    )

    registered_datasources = OrderedDict([('MNIST', mnist_tensorflow),
                                          ('FashionMNIST',
                                           fashion_mnist_tensorflow)])
else:
    from plato.datasources import (
        mnist,
        fashion_mnist,
        cifar10,
        cinic10,
        huggingface,
        pascal_voc,
        tiny_imagenet,
    )

    registered_datasources = OrderedDict([('MNIST', mnist),
                                          ('FashionMNIST', fashion_mnist),
                                          ('CIFAR10', cifar10),
                                          ('CINIC10', cinic10),
                                          ('HuggingFace', huggingface),
                                          ('PASCAL_VOC', pascal_voc),
                                          ('TinyImageNet', tiny_imagenet)])


def get():
    """Get the data source with the provided name."""
    datasource_name = Config().data.datasource

    logging.info("Data source: %s", Config().data.datasource)

    if Config().data.datasource == 'YOLO':
        from plato.datasources import yolo
        return yolo.DataSource()
    elif datasource_name in registered_datasources:
        dataset = registered_datasources[datasource_name].DataSource()
    else:
        raise ValueError('No such data source: {}'.format(datasource_name))

    return dataset
