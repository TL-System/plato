"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from trainers import (
    basic,
    scaffold,
    fedsarah,
)

from config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from trainers.mindspore import (
        basic as basic_mindspore, )

    registered_datasources = OrderedDict([
        ('basic', basic_mindspore.Trainer),
    ])
else:
    registered_trainers = OrderedDict([
        ('basic', basic.Trainer),
        ('scaffold', scaffold.Trainer),
        ('fedsarah', fedsarah.Trainer),
    ])


def get(client_id=0):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if Config().trainer.model_name == 'yolov5':
        from trainers import yolo
        return yolo.Trainer(client_id)
    elif Config().trainer.type == 'HuggingFace':
        from trainers import huggingface
        return huggingface.Trainer(client_id)
    elif trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name](client_id)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
