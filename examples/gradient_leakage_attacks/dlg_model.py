"""Obtaining models adapted from existing work's implementations.

An extra return object named `feature` is added in each model's forward function,
which will be used in the defense Soteria.
"""

from typing import Union

from nn import (
    lenet,
    resnet,
)

from plato.config import Config


def get(**kwargs: Union[str, dict]):
    """Get the model with the provided name."""
    model_name = (
        kwargs["model_name"] if "model_name" in kwargs else Config().trainer.model_name
    )

    if model_name == "lenet":
        return lenet.Model

    if model_name.split("_")[0] == "resnet":
        return resnet.get(model_name=model_name)

    # Set up model through plato's model library
    if Config().trainer.model_type == "vit":
        return None

    raise ValueError(f"No such model: {model_name}")
