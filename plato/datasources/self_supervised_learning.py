"""
A self-supervised learning dataset working as a wrapper to add the SSL data
transform to the datasource of Plato.

To allow SSL transform to use the desired parameters, one should place the
'data_transforms' sub-block under the 'algorithm' block in the config file.
"""

from lightly import transforms

from plato.config import Config
from plato.datasources import base
from plato.datasources import registry as datasources_registry

# The normalizations for different datasets
MNIST_NORMALIZE = {"mean": [0.1307], "std": [0.3081]}
FashionMNIST_NORMALIZE = {"mean": [0.1307], "std": [0.3081]}
EMNIST_NORMALIZE = {"mean": [0.5], "std": [0.5]}
CIFAR10_NORMALIZE = {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
CIFAR100_NORMALIZE = {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
STL10_NORMALIZE = {"mean": [0.4914, 0.4823, 0.4466], "std": [0.247, 0.243, 0.261]}

dataset_normalizations = {
    "MNIST": MNIST_NORMALIZE,
    "FashionMNIST": FashionMNIST_NORMALIZE,
    "EMNIST": EMNIST_NORMALIZE,
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
    transforms_node = Config().algorithm.data_transforms
    if hasattr(transforms_node, "_asdict"):
        transforms_config = transforms_node._asdict()
    elif isinstance(transforms_node, dict):
        transforms_config = dict(transforms_node)
    else:
        raise TypeError(
            "algorithm.data_transforms must be a mapping-like object "
            "with an '_asdict' method or behave as a dict."
        )

    # Set the data transform, which will be used as parameters to define the
    # SSL transform in registered_transforms
    data_transforms = {}
    if "train_transform" in transforms_config:
        raw_transform = transforms_config["train_transform"]
        if hasattr(raw_transform, "_asdict"):
            transform_config = raw_transform._asdict()
        elif isinstance(raw_transform, dict):
            transform_config = dict(raw_transform)
        else:
            raise TypeError(
                "train_transform configuration must provide an '_asdict' method "
                "or behave as a dict."
            )
        transform_name = transform_config["name"]
        raw_params = transform_config.get("parameters", {})
        if hasattr(raw_params, "_asdict"):
            transform_params = raw_params._asdict()
        elif isinstance(raw_params, dict):
            transform_params = dict(raw_params)
        else:
            raise TypeError(
                "train_transform.parameters must provide an '_asdict' method "
                "or behave as a dict."
            )

        # Get the data normalization for the datasource
        datasource_name = Config().data.datasource
        normalization_key = datasource_name
        if datasource_name == "Torchvision":
            if not hasattr(Config().data, "dataset_name"):
                raise ValueError(
                    "Torchvision datasource requires `dataset_name` to determine normalization."
                )
            normalization_key = Config().data.dataset_name

        if normalization_key not in dataset_normalizations:
            raise ValueError(
                f"No normalization defined for dataset: {normalization_key}"
            )

        transform_params["normalize"] = dataset_normalizations[normalization_key]
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
