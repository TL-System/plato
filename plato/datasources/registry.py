"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from plato.config import Config
from plato.datasources import (
    cinic10,
    feature,
    femnist,
    huggingface,
    lora,
    purchase,
    texas,
    tiny_imagenet,
    torchvision,
)

registered_datasources = {
    "HuggingFace": huggingface,
    "Torchvision": torchvision,
    "LoRA": lora,
    "CINIC10": cinic10,
    "Purchase": purchase,
    "Texas": texas,
    "TinyImageNet": tiny_imagenet,
    "Feature": feature,
}

registered_partitioned_datasources = {"FEMNIST": femnist}

_datasource_aliases = {
    "STL10": ("Torchvision", {"dataset_name": "STL10"}),
    "MNIST": ("Torchvision", {"dataset_name": "MNIST"}),
    "FashionMNIST": ("Torchvision", {"dataset_name": "FashionMNIST"}),
    "EMNIST": ("Torchvision", {"dataset_name": "EMNIST"}),
    "CIFAR10": ("Torchvision", {"dataset_name": "CIFAR10"}),
    "CIFAR100": ("Torchvision", {"dataset_name": "CIFAR100"}),
    "CelebA": ("Torchvision", {"dataset_name": "CelebA"}),
}


def get(client_id: int = 0, **kwargs):
    """Get the data source with the provided name."""
    datasource_name = (
        kwargs["datasource_name"]
        if "datasource_name" in kwargs
        else Config().data.datasource
    )

    logging.info("Data source: %s", datasource_name)

    if datasource_name in _datasource_aliases:
        target_name, extra_kwargs = _datasource_aliases[datasource_name]
        kwargs = {**extra_kwargs, **kwargs}
        datasource_name = target_name

    if datasource_name in registered_datasources:
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

    if datasource_name in _datasource_aliases:
        datasource_name = _datasource_aliases[datasource_name][0]

    if datasource_name in registered_datasources:
        input_shape = registered_datasources[datasource_name].DataSource.input_shape()
    elif datasource_name in registered_partitioned_datasources:
        input_shape = registered_partitioned_datasources[
            datasource_name
        ].DataSource.input_shape()
    else:
        raise ValueError(f"No such data source: {datasource_name}")

    return input_shape
