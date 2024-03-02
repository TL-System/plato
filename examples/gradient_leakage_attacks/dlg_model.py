"""Obtaining models adapted from existing work's implementations.

An extra return object named `feature` is added in each model's forward function,
which will be used in the defense Soteria.
"""

from plato.config import Config
from typing import Union
from transformers import ViTForImageClassification

from nn import (
    lenet,
    resnet,
)


def get(**kwargs: Union[str, dict]):
    """Get the model with the provided name."""
    model_name = (
        kwargs["model_name"] if "model_name" in kwargs else Config().trainer.model_name
    )

    if model_name == "lenet":
        return lenet.Model

    if model_name.split("_")[0] == "resnet":
        return resnet.get(model_name=model_name)
    
    if model_name == "vit":
        return ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224-in21k", num_labels=Config().parameters.model.num_classes
                )

    raise ValueError(f"No such model: {model_name}")
