"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging

from plato.config import Config

if hasattr(Config().trainer, "use_mindspore"):
    from plato.trainers.mindspore import basic as basic_mindspore

    registered_trainers = {"basic": basic_mindspore.Trainer}

elif hasattr(Config().trainer, "use_tensorflow"):
    from plato.trainers.tensorflow import basic as basic_tensorflow

    registered_trainers = {"basic": basic_tensorflow.Trainer}
else:
    from plato.trainers import (
        basic,
        diff_privacy,
        pascal_voc,
        gan,
    )

    registered_trainers = {
        "basic": basic.Trainer,
        "timm_basic": basic.TrainerWithTimmScheduler,
        "diff_privacy": diff_privacy.Trainer,
        "pascal_voc": pascal_voc.Trainer,
        "gan": gan.Trainer,
    }


def get(model=None):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if Config().trainer.model_name == "yolov5":
        from plato.trainers import yolov5

        return yolov5.Trainer()
    elif Config().trainer.type == "HuggingFace":
        from plato.trainers import huggingface

        return huggingface.Trainer(model)
    elif trainer_name in registered_trainers:
        return registered_trainers[trainer_name](model)
    else:
        raise ValueError(f"No such trainer: {trainer_name}")
