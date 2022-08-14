"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
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
        torch_hub,
        huggingface,
    )

    registered_models = {
        "lenet5": lenet5.Model,
        "dcgan": dcgan.Model,
        "multilayer": multilayer.Model,
    }

    registered_factories = {
        "resnet": resnet.Model,
        "vgg": vgg.Model,
        "torch_hub": torch_hub.Model,
        "huggingface": huggingface.Model,
    }


def get():
    """Get the model with the provided name."""
    model_name = Config().trainer.model_name
    model_type = (
        Config().trainer.model_type
        if hasattr(Config().trainer, "model_type")
        else model_name.split("_")[0]
    )
    if hasattr(Config().parameters, "model"):
        model_params = Config().parameters.model._asdict()
    else:
        model_params = {}

    if model_type in registered_models:
        registered_model = registered_models[model_type]
        return registered_model(**model_params)

    if model_type in registered_factories:
        return registered_factories[model_type].get(
            model_name=model_name, **model_params
        )

    # The YOLOv5 model needs special handling as it needs to import third-party packages
    if model_name == "yolov5":
        from plato.models import yolov5

        return yolov5.Model(**model_params)

    raise ValueError(f"No such model: {model_name}")
