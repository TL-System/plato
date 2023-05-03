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
        basic_personalized,
        diff_privacy,
        pascal_voc,
        gan,
    )

    registered_trainers = {
        "basic": basic.Trainer,
        "basic_personalized": basic_personalized.Trainer,
        "timm_basic": basic.TrainerWithTimmScheduler,
        "diff_privacy": diff_privacy.Trainer,
        "pascal_voc": pascal_voc.Trainer,
        "gan": gan.Trainer,
    }


def get(model=None, callbacks=None, **kwargs):
    """Get the trainer with the provided name."""

    trainer_name = kwargs["type"] if "type" in kwargs else Config().trainer.type
    model_name = (
        kwargs["model_name"] if "model_name" in kwargs else Config().trainer.model_name
    )

    logging.info("Trainer: %s", trainer_name)

    if model_name == "yolov5":
        from plato.trainers import yolov5

        return yolov5.Trainer()
    elif trainer_name == "HuggingFace":
        from plato.trainers import huggingface

        return huggingface.Trainer(model=model, callbacks=callbacks)
    elif trainer_name in registered_trainers:
        return registered_trainers[trainer_name](model=model, callbacks=callbacks)
    else:
        raise ValueError(f"No such trainer: {trainer_name}")
