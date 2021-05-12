"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
from collections import OrderedDict

from models import (lenet5, resnet, wideresnet, vgg, unet)

from config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from models.mindspore import (
        lenet5 as lenet5_mindspore, )

    registered_models = OrderedDict([
        ('lenet5', lenet5_mindspore.Model),
    ])
else:
    registered_models = OrderedDict([('lenet5', lenet5.Model),
                                     ('resnet', resnet.Model),
                                     ('vgg', vgg.Model),
                                     ('wideresnet', wideresnet.Model),
                                     ('unet', unet.Model)])


def get():
    """Get the model with the provided name."""
    model_name = Config().trainer.model_name
    model_type = model_name.split('_')[0]
    model = None

    if model_name == 'yolov5':
        from models import yolo
        return yolo.Model.get_model()

    if model_name == 'HuggingFace_CausalLM':
        from transformers import AutoModelForCausalLM
        model_checkpoint = Config.trainer.model_checkpoint
        return AutoModelForCausalLM.from_pretrained(model_checkpoint)

    else:
        for name, registered_model in registered_models.items():
            if name.startswith(model_type):
                model = registered_model.get_model(model_name)

    if model is None:
        raise ValueError('No such model: {}'.format(model_name))

    return model
