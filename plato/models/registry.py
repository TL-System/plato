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
        resnet,
        torch_hub,
        wideresnet,
        inceptionv3,
        googlenet,
        vgg,
        unet,
        alexnet,
        squeezenet,
        hybrid,
        efficientnet,
        regnet,
        dcgan,
        multilayer,
    )

    registered_models = OrderedDict(
        [
            ("lenet5", lenet5.Model),
            ("wideresnet", wideresnet.Model),
            ("inceptionv3", inceptionv3.Model),
            ("googlenet", googlenet.Model),
            ("vgg", vgg.Model),
            ("unet", unet.Model),
            ("alexnet", alexnet.Model),
            ("squeezenet", squeezenet.Model),
            ("hybrid", hybrid.Model),
            ("efficientnet", efficientnet.Model),
            ("regnet", regnet.Model),
            ("dcgan", dcgan.Model),
            ("multilayer", multilayer.Model),
        ]
    )

    registered_factories = OrderedDict(
        [
            ("resnet", resnet.Model),
            ("torch_hub", torch_hub.Model),
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

    if model_type == "HuggingFace_CausalLM":
        from transformers import AutoModelForCausalLM, AutoConfig

        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        return AutoModelForCausalLM.from_pretrained(
            model_name, config=config, cache_dir="./models/huggingface"
        )

    else:
        for name, registered_model in registered_models.items():
            if name.startswith(model_type):
                num_classes = (
                    Config().trainer.num_classes
                    if hasattr(Config().trainer, "num_classes")
                    else None
                )
                model = registered_model(
                    num_classes=num_classes,
                )

        if model is None:
            for name, registered_factory in registered_factories.items():
                if name.startswith(model_type):
                    num_classes = (
                        Config().trainer.num_classes
                        if hasattr(Config().trainer, "num_classes")
                        else None
                    )
                    pretrained = (
                        Config().trainer.pretrained
                        if hasattr(Config().trainer, "pretrained")
                        else False
                    )
                    model = registered_factory.get(
                        model_name=model_name,
                        num_classes=num_classes,
                        pretrained=pretrained,
                    )

    if model is None:
        raise ValueError(f"No such model: {model_name}")

    return model
