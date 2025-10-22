"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from plato.config import Config
from plato.trainers import (
    basic,
    composable,
    diff_privacy,
    gan,
    mlx,
    split_learning,
)

registered_trainers = {
    "composable": composable.ComposableTrainer,
    "basic": basic.Trainer,
    "timm_basic": basic.TrainerWithTimmScheduler,
    "diff_privacy": diff_privacy.Trainer,
    "gan": gan.Trainer,
    "split_learning": split_learning.Trainer,
    "mlx": mlx.ComposableMLXTrainer,
}


def _resolve_trainer_name(trainer_config) -> str:
    """Resolve trainer type supporting framework shortcuts."""
    trainer_type = getattr(trainer_config, "type", None)
    framework = getattr(trainer_config, "framework", "")

    if not trainer_type and framework:
        if framework.lower() == "mlx":
            return "mlx"
    return trainer_type


def get(model=None, callbacks=None):
    """Get the trainer with the provided name."""
    config = Config().trainer
    trainer_name = _resolve_trainer_name(config)
    logging.info("Trainer: %s", trainer_name)

    if trainer_name == "HuggingFace":
        from plato.trainers import huggingface

        return huggingface.Trainer(model=model, callbacks=callbacks)

    elif trainer_name == "self_supervised_learning":
        from plato.trainers import self_supervised_learning

        return self_supervised_learning.Trainer(model=model, callbacks=callbacks)
    elif trainer_name in registered_trainers:
        return registered_trainers[trainer_name](model=model, callbacks=callbacks)
    else:
        raise ValueError(f"No such trainer: {trainer_name}")
