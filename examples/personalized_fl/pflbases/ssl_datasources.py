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

    # Get the transforms details set in the config file
    transforms_config = Config().algorithm.data_transforms._asdict()

    # Set the data transform, which will be used as parameters to define the
    # ssl transform in registered_transforms
    data_transforms = {}
    if "train_transform" in transforms_config:
        transform_config = transforms_config["train_transform"]._asdict()
        transform_name = transform_config["name"]
        transform_params = transform_config["parameters"]._asdict()

        # Get the data normalization for the datasource
        datasource_name = Config().data.datasource
        transform_params["normalize"] = visual_augmentations.datasets_normalization[
            datasource_name
        ]
        # Get the SSL transform
        if transform_name in registered_transforms:
            dataset_transform = registered_transforms[transform_name](
                **transform_params
            )
        else:
            raise ValueError(f"No such data source: {transform_name}")

        # Insert the obtained transform to the data_transforms
        # to be used by the datasource of Plato to get the train/test set
        data_transforms.update({"train_transform": dataset_transform})

    return data_transforms


# pylint: disable=W0223
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
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
