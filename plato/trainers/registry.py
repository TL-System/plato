"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.trainers import (
    basic,
    scaffold,
    fedsarah,
    split_learning,
    pascal_voc,
)

from plato.config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from plato.trainers.mindspore import (
        basic as basic_mindspore, )

    registered_trainers = OrderedDict([
        ('basic', basic_mindspore.Trainer),
    ])
else:
    registered_trainers = OrderedDict([
        ('basic', basic.Trainer),
        ('scaffold', scaffold.Trainer),
        ('fedsarah', fedsarah.Trainer),
        ('split_learning', split_learning.Trainer),
        ('pascal_voc', pascal_voc.Trainer),
    ])


def get(client_id=0, model=None):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if Config().trainer.model_name == 'yolov5':
        from plato.trainers import yolo
        return yolo.Trainer(client_id)
    elif Config().trainer.type == 'HuggingFace':
        from plato.trainers import huggingface
        return huggingface.Trainer(model)
    elif trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name](model)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
