"""
The transform factory.
"""

from lightly.transforms import *

from plato.config import Config


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


def get(client_id: int = 0, **kwargs):
    """Get the data source with the provided name."""
    data_transform_name = (
        kwargs["data_transform_name"]
        if "data_transform_name" in kwargs
        else Config().data.train_transform
    )
    data_transform_params = (
        kwargs["data_transform_params"]
        if "data_transform_params" in kwargs
        else Config().parameters.train_transform
    )

    if data_transform_name in registered_transforms:
        dataset_transform = registered_transforms[data_transform_name](
            **data_transform_params
        )
    else:
        raise ValueError(f"No such data source: {data_transform_name}")

    return dataset_transform
