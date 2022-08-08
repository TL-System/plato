"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
from collections import OrderedDict

from plato.config import Config

if hasattr(Config().trainer, "use_mindspore"):
    from plato.models.mindspore import (
        lenet5 as lenet5_mindspore,
    )

    registered_models = OrderedDict(
        [
            ("lenet5", lenet5_mindspore.Model),
        ]
    )
elif hasattr(Config().trainer, "use_tensorflow"):
    from plato.models.tensorflow import (
        lenet5 as lenet5_tensorflow,
    )

    registered_models = OrderedDict(
        [
            ("lenet5", lenet5_tensorflow.Model),
        ]
    )
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

    registered_models = OrderedDict(
        [
            ("lenet5", lenet5.Model),
            ("dcgan", dcgan.Model),
            ("multilayer", multilayer.Model),
        ]
    )

    registered_factories = OrderedDict(
        [
            ("resnet", resnet.Model),
            ("vgg", vgg.Model),
            ("torch_hub", torch_hub.Model),
            ("huggingface", huggingface.Model),
        ]
    )


def get():
    """Get the model with the provided name."""
    model_name = Config().trainer.model_name
    model_type = (
        Config().trainer.model_type
        if hasattr(Config().trainer, "model_type")
        else model_name.split("_")[0]
    )
    model = None

    if model_name == "yolov5":
        from plato.models import yolo

        if hasattr(Config().trainer, "model_config"):
            return yolo.Model(Config().trainer.model_config, Config().data.num_classes)
        else:
            return yolo.Model("yolov5s.yaml", Config().data.num_classes)
    else:
        for name, registered_model in registered_models.items():
            if name.startswith(model_type):
                num_classes = (
                    Config().trainer.num_classes
                    if hasattr(Config().trainer, "num_classes")
                    else 10
                )
                model = registered_model(num_classes=num_classes)

        if model is None:
            for name, registered_factory in registered_factories.items():
                if name.startswith(model_type):
                    num_classes = (
                        Config().trainer.num_classes
                        if hasattr(Config().trainer, "num_classes")
                        else None
                    )
                    model = registered_factory.get(
                        model_name=model_name,
                        num_classes=num_classes,
                    )

    if model is None:
        raise ValueError(f"No such model: {model_name}")

    return model
