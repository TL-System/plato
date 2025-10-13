"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from typing import Any, Dict, TypedDict, cast

from plato.config import Config
from plato.models import (
    cnn_encoder,
    dcgan,
    general_multilayer,
    huggingface,
    lenet5,
    multilayer,
    resnet,
    torch_hub,
    vgg,
    vit,
)

registered_models = {
    "lenet5": lenet5.Model,
    "dcgan": dcgan.Model,
    "multilayer": multilayer.Model,
}

registered_factories = {
    "resnet": resnet.Model,
    "vgg": vgg.Model,
    "cnn_encoder": cnn_encoder.Model,
    "general_multilayer": general_multilayer.Model,
    "torch_hub": torch_hub.Model,
    "huggingface": huggingface.Model,
    "vit": vit.Model,
}


class ModelKwargs(TypedDict, total=False):
    model_name: str
    model_type: str
    model_params: Dict[str, Any]


def get(**kwargs: Any) -> Any:
    """Get the model with the provided name."""
    config = Config()

    # Get model name
    model_name: str = ""
    if "model_name" in kwargs:
        model_name = cast(str, kwargs["model_name"])
    elif hasattr(config, "trainer"):
        trainer = getattr(config, "trainer")
        if hasattr(trainer, "model_name"):
            model_name = getattr(trainer, "model_name")

    # Get model type
    model_type: str = ""
    if "model_type" in kwargs:
        model_type = cast(str, kwargs["model_type"])
    elif hasattr(config, "trainer"):
        trainer = getattr(config, "trainer")
        if hasattr(trainer, "model_type"):
            model_type = getattr(trainer, "model_type")

    # If model_type is still empty, derive it from model_name
    if not model_type and model_name:
        model_type = model_name.split("_")[0]

    # Get model parameters
    model_params: Dict[str, Any] = {}
    if "model_params" in kwargs:
        model_params = cast(Dict[str, Any], kwargs["model_params"])
    elif hasattr(config, "parameters"):
        parameters = getattr(config, "parameters")
        if hasattr(parameters, "model"):
            model = getattr(parameters, "model")
            if hasattr(model, "_asdict"):
                model_params = model._asdict()

    if model_type in registered_models:
        registered_model = registered_models[model_type]
        return registered_model(**model_params)

    if model_type in registered_factories:
        return registered_factories[model_type].get(
            model_name=model_name, **model_params
        )

    raise ValueError(f"No such model: {model_name}")
