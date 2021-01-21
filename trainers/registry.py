"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from models.base import Model
from trainers import trainer, mistnet, adaptive_freezing, adaptive_sync
from config import Config

registered_trainers = {
    'basic': trainer,
    'mistnet': mistnet,
    'adaptive_freezing': adaptive_freezing,
    'adaptive_sync': adaptive_sync
}


def get(model: Model):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type

    if trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name].Trainer(model)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
