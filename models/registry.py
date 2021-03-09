"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
from collections import OrderedDict

from models import (
    lenet5,
    resnet,
    wideresnet,
    vgg,
    feedback_transformer,
)

from config import Config

registered_models = OrderedDict([
    ('lenet5', lenet5.Model),
    ('resnet_18', resnet.Model),
    ('resnet_34', resnet.Model),
    ('resnet_50', resnet.Model),
    ('resnet_101', resnet.Model),
    ('resnet_152', resnet.Model),
    ('vgg_11', vgg.Model),
    ('vgg_13', vgg.Model),
    ('vgg_16', vgg.Model),
    ('vgg_19', vgg.Model),
    ('wideresnet', wideresnet.Model),
    ('feedback_transformer', feedback_transformer.Model),
])

if Config().trainer.model == 'yolov5':
    from models import yolo
    registered_models += ('yolov5', yolo.Model)

if hasattr(Config().trainer, 'use_mindspore'):
    from models import lenet5_mindspore
    registered_models += ('lenet5_mindspore', lenet5_mindspore.Model)


def get(model_type):
    """Get the model with the provided type."""
    if model_type in registered_models:
        model = registered_models[model_type].get_model(model_type)
    else:
        raise ValueError('No such model: {}'.format(model_type))

    return model
