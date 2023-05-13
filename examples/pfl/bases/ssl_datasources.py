"""
Customize a flexible interface of datasource for personalized federated learning.
"""
import logging

from plato.datasources import base
from plato.datasources import registry as datasources_registry
from plato.config import Config

from bases import transform_registry


def get_transform(transform_config: dict):
    """Getting transform for the desired transform_type."""

    transform_name = transform_config["name"]
    transform_params = transform_config["parameters"]._asdict()
    defined_transform = transform_registry.get(
        data_transform_name=transform_name,
        data_transform_params=transform_params,
    )

    return defined_transform


def get_data_transforms(transforms_block: dict):
    """Obtaining train/test transforms for the corresponding data."""

    data_transforms = {}

    if "train_transform" in transforms_block:
        transform_config = transforms_block["train_transform"]._asdict()
        train_transform = get_transform(transform_config)
        logging.info("Data train transform: %s", transform_config["name"])
        data_transforms.update({"train_transform": train_transform})

    if "test_transform" in transforms_block:
        transform_config = transforms_block["test_transform"]._asdict()
        test_transform = get_transform(transform_config)
        logging.info("Data test transform: %s", transform_config["name"])
        data_transforms.update({"test_transform": test_transform})

    return data_transforms


class TransformedDataSource(base.DataSource):
    """A custom datasource receiving configuration of transform as the
    input to define the datasource.
    """

    def __init__(self, transforms_block: dict = None):
        super().__init__()
        # use the default config of the transform when nothing
        # is provided.
        if transforms_block is None:
            transforms_block = Config().algorithm.data_transforms._asdict()

        data_transforms = get_data_transforms(transforms_block)

        self.datasource = datasources_registry.get(**data_transforms)
        self.trainset = self.datasource.trainset
        self.testset = self.datasource.testset

    def num_train_examples(self):
        return self.datasource.num_train_examples()

    def num_test_examples(self):
        return self.datasource.num_test_examples()
