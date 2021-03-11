"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from models.base import Model
from trainers import (
    trainer,
    scaffold,
    fedsarah,
)

from config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from trainers.mindspore import (
        trainer as trainer_mindspore, )

    registered_datasources = OrderedDict([
        ('basic', trainer_mindspore),
    ])
else:
    registered_trainers = OrderedDict([
        ('basic', trainer),
        ('scaffold', scaffold),
        ('fedsarah', fedsarah),
    ])

    if Config().trainer.model == 'yolov5':
        from trainers import yolo
        yolo_trainers = {'yolov5': yolo.Trainer}

        registered_trainers = OrderedDict(
            list(registered_trainers.items()) + list(yolo_trainers.items()))


def get(model: Model, client_id=0):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name](model,
                                                               client_id)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
