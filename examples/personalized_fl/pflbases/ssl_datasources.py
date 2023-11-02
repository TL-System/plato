"""
A base datasource for self-supervised learning.
"""
from lightly import transforms

from plato.datasources import base
from plato.datasources import registry as datasources_registry
from plato.config import Config
from plato.utils import visual_augmentations


# All transforms for different SSL algorithms
registered_transforms = {
    "SimCLR": transforms.SimCLRTransform,
    "DINO": transforms.DINOTransform,
    "MAE": transforms.MAETransform,
    "MoCoV1": transforms.MoCoV1Transform,
    "MoCoV2": transforms.MoCoV2Transform,
    "MSN": transforms.MSNTransform,
    "PIRL": transforms.PIRLTransform,
    "SimSiam": transforms.SimSiamTransform,
    "SMoG": transforms.SMoGTransform,
    "SwaV": transforms.SwaVTransform,
    "VICReg": transforms.VICRegTransform,
    "VICRegL": transforms.VICRegLTransform,
    "FastSiam": transforms.FastSiamTransform,
}


def get_transforms():
    """Obtains train/test transforms for the corresponding data."""
    transforms_config = Config().algorithm.data_transforms._asdict()

    data_transforms = {}
    if "train_transform" in transforms_config:
        transform_config = transforms_config["train_transform"]._asdict()
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
    """A base datasource to define the DataSource for self-supervised
    learning."""

    def __init__(self):
        super().__init__()

        # Get the transforms for the data
        data_transforms = get_transforms()

        self.datasource = datasources_registry.get(**data_transforms)
        self.trainset = self.datasource.trainset
        self.testset = self.datasource.testset

    def num_train_examples(self):
        return self.datasource.num_train_examples()

    def num_test_examples(self):
        return self.datasource.num_test_examples()
