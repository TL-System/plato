"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from models import lenet5, resnet, wideresnet, vgg, feedback_transformer
from config import Config

registered_models = [
    lenet5.Model, resnet.Model, wideresnet.Model, vgg.Model,
    feedback_transformer.Model
]

if Config().trainer.model == 'yolov5':
    from models import yolo
    registered_models += [yolo.Model]

if hasattr(Config().trainer, 'use_mindspore'):
    from models import lenet5_mindspore
    registered_models += [lenet5_mindspore.Model]


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
