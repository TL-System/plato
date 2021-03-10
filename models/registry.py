"""
The registry for machine learning models.

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

if hasattr(Config().trainer, 'use_mindspore'):
    from models import lenet5_mindspore
    
    registered_models = OrderedDict([
        ('lenet5_mindspore', lenet5_mindspore.Model),
    ])
else:
    registered_models = OrderedDict([
        ('lenet5', lenet5.Model),
        ('resnet', resnet.Model),
        ('vgg', vgg.Model),
        ('wideresnet', wideresnet.Model),
        ('feedback_transformer', feedback_transformer.Model),
    ])

    if Config().trainer.model == 'yolov5':
        from models import yolo
        registered_models += ('yolov5', yolo.Model)


def get():
    """Get the model with the provided name."""
    model_name = Config().trainer.model
    model_type = model_name.split('_')[0]
    model = None

    for name, registered_model in registered_models.items():
        if name.startswith(model_type):
            model = registered_model.get_model(model_name)

    if model is None:
        raise ValueError('No such model: {}'.format(model_name))
    else:
        return model
