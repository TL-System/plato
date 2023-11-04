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
    from plato.trainers import basic, diff_privacy, pascal_voc, gan, split_learning

    registered_trainers = {
        "basic": basic.Trainer,
        "timm_basic": basic.TrainerWithTimmScheduler,
        "diff_privacy": diff_privacy.Trainer,
        "pascal_voc": pascal_voc.Trainer,
        "gan": gan.Trainer,
        "split_learning": split_learning.Trainer,
    }


def get(model=None, callbacks=None):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if Config().trainer.model_name == "yolov8":
        from plato.trainers import yolov8

        return yolov8.Trainer()
    elif Config().trainer.type == "HuggingFace":
        from plato.trainers import huggingface

        return huggingface.Trainer(model=model, callbacks=callbacks)

    elif Config().trainer.type == "self_supervised_learning":
        from plato.trainers import self_supervised_learning

        return self_supervised_learning.Trainer(model=model, callbacks=callbacks)
    elif trainer_name in registered_trainers:
        return registered_trainers[trainer_name](model=model, callbacks=callbacks)
    else:
        raise ValueError(f"No such trainer: {trainer_name}")
