"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from models import lenet5_pytorch, resnet_pytorch, wideresnet_pytorch, vgg_pytorch
from models import lenet5_mindspore

registered_models = [
    lenet5_pytorch.Model, resnet_pytorch.Model, wideresnet_pytorch.Model,
    vgg_pytorch.Model, lenet5_mindspore.Model
]


def get(model_name):
    """Get the model with the provided name."""
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            model = registered_model.get_model_from_name(model_name)
            break

    if model is None:
        raise ValueError('No such model: {}'.format(model_name))

    return model
