"""
Customize callbacks for self-supervised trainer performing separate optimization of 
the global model and personalized models. Typical approaches are SimCLR, BYOL.

Therefore,
At the end of normal local training, the model will be saved for subsequent usage.
"""

import logging

from plato.config import Config

from plato.callbacks import trainer as trainer_callbacks
from plato.utils.filename_formatter import NameFormatter


class ModelStatusCallback(trainer_callbacks.TrainerCallback):
    """A trainer callback to record the model,
        1). only at the end of each round of local update.
    ."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 3)."""

        if not trainer.personalized_learning:
            save_location = trainer.get_checkpoint_dir_path()
            filename = NameFormatter.get_format_name(
                client_id=trainer.client_id,
                model_name=Config().trainer.model_name,
                round_n=trainer.current_round,
                epoch_n=None,
                run_id=None,
                prefix=None,
                ext="pth",
            )
            trainer.save_model(filename=filename, location=save_location)
