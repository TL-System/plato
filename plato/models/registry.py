"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
from typing import Union

from plato.config import Config

if hasattr(Config().trainer, "use_mindspore"):
    from plato.models.mindspore import lenet5 as lenet5_mindspore

    registered_models = {"lenet5": lenet5_mindspore.Model}
elif hasattr(Config().trainer, "use_tensorflow"):
    from plato.models.tensorflow import lenet5 as lenet5_tensorflow

    registered_models = {"lenet5": lenet5_tensorflow.Model}
else:
    from plato.models import (
        lenet5,
        dcgan,
        multilayer,
        resnet,
        vgg,
        cnn_encoder,
        general_multilayer,
        torch_hub,
        huggingface,
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


def get(**kwargs: Union[str, dict]):
    """Get the model with the provided name."""
    model_name = (
        kwargs["model_name"] if "model_name" in kwargs else Config().trainer.model_name
    )

    model_type = (
        kwargs["model_type"]
        if "model_type" in kwargs
        else (
            Config().trainer.model_type
            if hasattr(Config().trainer, "model_type")
            else model_name.split("_")[0]
        )
    )

    model_params = (
        kwargs["model_params"]
        if "model_params" in kwargs
        else (
            Config().parameters.model._asdict()
            if hasattr(Config().parameters, "model")
            else {}
        )
    )

    if model_type in registered_models:
        registered_model = registered_models[model_type]
        return registered_model(**model_params)

    if model_type in registered_factories:
        return registered_factories[model_type].get(
            model_name=model_name, **model_params
        )

    # The YOLOv8 model needs special handling as it needs to import third-party packages
    if model_name == "yolov8":
        from plato.models import yolov8

        return yolov8.Model.get()

    raise ValueError(f"No such model: {model_name}")
