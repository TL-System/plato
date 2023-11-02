"""
A base datasource for self-supervised learning.
"""
import logging

from lightly.transforms import *

from plato.datasources import base
from plato.datasources import registry as datasources_registry
from plato.config import Config
from plato.utils import visual_augmentations


registered_transforms = {
    "SimCLR": SimCLRTransform,
    "DINO": DINOTransform,
    "MAE": MAETransform,
    "MoCoV1": MoCoV1Transform,
    "MoCoV2": MoCoV2Transform,
    "MSN": MSNTransform,
    "PIRL": PIRLTransform,
    "SimSiam": SimSiamTransform,
    "SMoG": SMoGTransform,
    "SwaV": SwaVTransform,
    "VICReg": VICRegTransform,
    "VICRegL": VICRegLTransform,
    "FastSiam": FastSiamTransform,
}


def get_transforms(transforms_block: dict):
    """Obtains train/test transforms for the corresponding data."""

    data_transforms = {}

    if "train_transform" in transforms_block:
        transform_config = transforms_block["train_transform"]._asdict()
        transform_name = transform_config["name"]
        transform_params = transform_config["parameters"]._asdict()

        datasource_name = Config().data.datasource

        transform_params["normalize"] = visual_augmentations.datasets_normalization[
            datasource_name
        ]
        if transform_name in registered_transforms:
            dataset_transform = registered_transforms[transform_name](
                **transform_params
            )
        else:
            raise ValueError(f"No such data source: {transform_name}")

        data_transforms.update({"train_transform": dataset_transform})

    return data_transforms


class SSLDataSource(base.DataSource):
    """A custom datasource receiving configuration of transform as the
    input to define the datasource.
    """

    def __init__(self):
        super().__init__()
        # Use the transform set in the config file.
        transforms_block = Config().algorithm.data_transforms._asdict()

        data_transforms = get_transforms(transforms_block)

        self.datasource = datasources_registry.get(**data_transforms)
        self.trainset = self.datasource.trainset
        self.testset = self.datasource.testset

    def num_train_examples(self):
        return self.datasource.num_train_examples()

    def num_test_examples(self):
        return self.datasource.num_test_examples()
