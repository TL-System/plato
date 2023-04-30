"""
The transform factory.
"""

from lightly.transforms import *

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


def extract_normalization(datasource):
    """Get the normalization for datasource."""
    data_norm = visual_augmentations.datasets_normalization[datasource]
    return data_norm


def get(client_id: int = 0, **kwargs):
    """Get the data source with the provided name."""

    datasource_name = (
        kwargs["datasource_name"]
        if "datasource_name" in kwargs
        else Config().data.datasource
    )

    data_transform_name = (
        kwargs["data_transform_name"]
        if "data_transform_name" in kwargs
        else Config().data.train_transform
    )
    data_transform_params = (
        kwargs["data_transform_params"]
        if "data_transform_params" in kwargs
        else Config().parameters.train_transform._asdict()
        if hasattr(Config().parameters, "train_transform")
        else {}
    )

    data_transform_params["normalize"] = extract_normalization(datasource_name)

    if data_transform_name in registered_transforms:
        dataset_transform = registered_transforms[data_transform_name](
            **data_transform_params
        )
    else:
        raise ValueError(f"No such data source: {data_transform_name}")

    return dataset_transform
