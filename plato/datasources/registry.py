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
    registered_partitioned_datasources = OrderedDict()

elif hasattr(Config().trainer, 'use_tensorflow'):
    from plato.datasources.tensorflow import (
        mnist as mnist_tensorflow,
        fashion_mnist as fashion_mnist_tensorflow,
    )

    registered_datasources = OrderedDict([('MNIST', mnist_tensorflow),
                                          ('FashionMNIST',
                                           fashion_mnist_tensorflow)])

elif hasattr(Config.data, 'use_multimodal'):
    from plato.datasources.multimodal import kinetics, gym, flickr30k_entities, referitgame

    registered_datasources = OrderedDict([
        ('kinetics700', kinetics),
        ('kinetics400', kinetics),
        ('Gym', gym),
        ('Flickr30E', flickr30k_entities),
        ('Referitgame', referitgame),
    ])
    registered_partitioned_datasources = OrderedDict()

else:
    from plato.datasources import (
        mnist,
        fashion_mnist,
        cifar10,
        cinic10,
        huggingface,
        pascal_voc,
        tiny_imagenet,
        femnist,
        feature,
    )

    registered_datasources = OrderedDict([('MNIST', mnist),
                                          ('FashionMNIST', fashion_mnist),
                                          ('CIFAR10', cifar10),
                                          ('CINIC10', cinic10),
                                          ('HuggingFace', huggingface),
                                          ('PASCAL_VOC', pascal_voc),
                                          ('TinyImageNet', tiny_imagenet),
                                          ('Feature', feature)])

    registered_partitioned_datasources = OrderedDict([('FEMNIST', femnist)])


def get(client_id=0):
    """Get the data source with the provided name."""
    datasource_name = Config().data.datasource

    logging.info("Data source: %s", Config().data.datasource)

    if Config().data.datasource == 'YOLO':
        from plato.datasources import yolo
        return yolo.DataSource()
    elif datasource_name in registered_datasources:
        dataset = registered_datasources[datasource_name].DataSource()
    elif datasource_name in registered_partitioned_datasources:
        dataset = registered_partitioned_datasources[
            datasource_name].DataSource(client_id)
    else:
        raise ValueError('No such data source: {}'.format(datasource_name))

    return dataset


def get_input_shape():
    """Get the input shape of data source with the provided name."""
    datasource_name = Config().data.datasource

    logging.info("Data source: %s", Config().data.datasource)

    if Config().data.datasource == 'YOLO':
        from plato.datasources import yolo
        return yolo.DataSource.input_shape()
    elif datasource_name in registered_datasources:
        input_shape = registered_datasources[
            datasource_name].DataSource.input_shape()
    elif datasource_name in registered_partitioned_datasources:
        input_shape = registered_partitioned_datasources[
            datasource_name].DataSource.input_shape()
    else:
        raise ValueError('No such data source: {}'.format(datasource_name))

    return input_shape
