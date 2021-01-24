"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging

from models.base import Model
from trainers import trainer, mistnet, adaptive_freezing, adaptive_sync, fednova

from config import Config

registered_trainers = {
    'basic': trainer,
    'mistnet': mistnet,
    'adaptive_freezing': adaptive_freezing,
    'adaptive_sync': adaptive_sync,
    'fednova': fednova
}

if Config().trainer.use_mindspore:
    from trainers import trainer_mindspore
    mindspore_trainers = {'basic_mindspore': trainer_mindspore}
    registered_trainers = dict(
        list(registered_trainers.items()) + list(mindspore_trainers.items()))


def get(model: Model, client_id=0):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type

    if trainer_name in registered_trainers:
        logging.info("Trainer: %s", trainer_name)
        registered_trainer = registered_trainers[trainer_name].Trainer(
            model, client_id)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
