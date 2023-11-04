"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from plato.config import Config

if hasattr(Config().trainer, "use_mindspore"):
    from plato.datasources.mindspore import mnist as mnist_mindspore

    registered_datasources = {"MNIST": mnist_mindspore}
    registered_partitioned_datasources = {}

elif hasattr(Config().trainer, "use_tensorflow"):
    from plato.datasources.tensorflow import (
        mnist as mnist_tensorflow,
        fashion_mnist as fashion_mnist_tensorflow,
    )

    registered_datasources = {
        "MNIST": mnist_tensorflow,
        "FashionMNIST": fashion_mnist_tensorflow,
    }

else:
    from plato.datasources import (
        mnist,
        fashion_mnist,
        emnist,
        cifar10,
        cifar100,
        cinic10,
        purchase,
        texas,
        huggingface,
        pascal_voc,
        tiny_imagenet,
        femnist,
        feature,
        qoenflx,
        celeba,
        stl10,
    )

    registered_datasources = {
        "MNIST": mnist,
        "FashionMNIST": fashion_mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
        "CIFAR100": cifar100,
        "CINIC10": cinic10,
        "Purchase": purchase,
        "Texas": texas,
        "HuggingFace": huggingface,
        "PASCAL_VOC": pascal_voc,
        "TinyImageNet": tiny_imagenet,
        "Feature": feature,
        "QoENFLX": qoenflx,
        "CelebA": celeba,
        "STL10": stl10,
    }

    registered_partitioned_datasources = {"FEMNIST": femnist}


def get(client_id: int = 0, **kwargs):
    """Get the data source with the provided name."""
    datasource_name = (
        kwargs["datasource_name"]
        if "datasource_name" in kwargs
        else Config().data.datasource
    )

    logging.info("Data source: %s", datasource_name)

    if datasource_name == "kinetics700":
        from plato.datasources import kinetics

        return kinetics.DataSource(**kwargs)

    if datasource_name == "Gym":
        from plato.datasources import gym

        return gym.DataSource(**kwargs)

    if datasource_name == "Flickr30KE":
        from plato.datasources import flickr30k_entities

        return flickr30k_entities.DataSource(**kwargs)

    if datasource_name == "ReferItGame":
        from plato.datasources import referitgame

        return referitgame.DataSource(**kwargs)

    if datasource_name == "COCO":
        from plato.datasources import coco

        return coco.DataSource(**kwargs)

    if datasource_name == "YOLOv8":
        from plato.datasources import yolov8

        return yolov8.DataSource(**kwargs)
    elif datasource_name in registered_datasources:
        dataset = registered_datasources[datasource_name].DataSource(**kwargs)
    elif datasource_name in registered_partitioned_datasources:
        dataset = registered_partitioned_datasources[datasource_name].DataSource(
            client_id, **kwargs
        )
    else:
        raise ValueError(f"No such data source: {datasource_name}")

    return dataset


def get_input_shape():
    """Get the input shape of data source with the provided name."""
    datasource_name = Config().data.datasource

    logging.info("Data source: %s", Config().data.datasource)
    if Config().data.datasource == "YOLO":
        from plato.datasources import yolo

        return yolo.DataSource.input_shape()
    elif datasource_name in registered_datasources:
        input_shape = registered_datasources[datasource_name].DataSource.input_shape()
    elif datasource_name in registered_partitioned_datasources:
        input_shape = registered_partitioned_datasources[
            datasource_name
        ].DataSource.input_shape()
    else:
        raise ValueError(f"No such data source: {datasource_name}")

    return input_shape
