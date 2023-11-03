"""
A self-supervised learning dataset working as a wrapper to add the SSL data
transform to the datasource of Plato.

To allow SSL transform to use the desired parameters, one should place the
'data_transforms' sub-block under the 'algorithm' block in the config file. 
"""
from lightly import transforms

from plato.datasources import base
from plato.datasources import registry as datasources_registry
from plato.config import Config


# The normalizations for different datasets
MNIST_NORMALIZE = {"mean": [0.1307], "std": [0.3081]}
FashionMNIST_NORMALIZE = {"mean": [0.1307], "std": [0.3081]}
CIFAR10_NORMALIZE = {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
CIFAR100_NORMALIZE = {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
STL10_NORMALIZE = {"mean": [0.4914, 0.4823, 0.4466], "std": [0.247, 0.243, 0.261]}

dataset_normalizations = {
    "MNIST": MNIST_NORMALIZE,
    "FashionMNIST": FashionMNIST_NORMALIZE,
    "CIFAR10": CIFAR10_NORMALIZE,
    "CIFAR100": CIFAR100_NORMALIZE,
    "IMAGENET": IMAGENET_NORMALIZE,
    "STL10": STL10_NORMALIZE,
}


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
    """Obtain train/test transforms for the corresponding data."""

    # Get the transforms details set in the config file
    transforms_config = Config().algorithm.data_transforms._asdict()

    # Set the data transform, which will be used as parameters to define the
    # SSL transform in registered_transforms
    data_transforms = {}
    if "train_transform" in transforms_config:
        transform_config = transforms_config["train_transform"]._asdict()
        transform_name = transform_config["name"]
        transform_params = transform_config["parameters"]._asdict()

        # Get the data normalization for the datasource
        datasource_name = Config().data.datasource
        transform_params["normalize"] = dataset_normalizations[datasource_name]
        # Get the SSL transform
        if transform_name in registered_transforms:
            dataset_transform = registered_transforms[transform_name](
                **transform_params
            )
        else:
            raise ValueError(f"No such data source: {transform_name}")

        # Insert the obtained transform to the data_transforms.
        # It is used by the datasource of Plato to get the train/test set.
        data_transforms.update({"train_transform": dataset_transform})

    return data_transforms


# pylint: disable=abstract-method
class SSLDataSource(base.DataSource):
    """
    An SSL datasource to define the dataSource for self-supervised learning.
    """

    def __init__(self):
        super().__init__()

        # Get the transforms for the data
        data_transforms = get_transforms()

        self.datasource = datasources_registry.get(**data_transforms)
        self.trainset = self.datasource.trainset
        self.testset = self.datasource.testset
