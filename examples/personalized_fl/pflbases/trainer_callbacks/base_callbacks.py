"""
Callbacks for the personalized trainer.

The `PersonalizedModelCallback` is the base callback to 
record the personalized model.

The user is desired to inherit from these two basic callbacks to create
customized ones.
"""

import os
import logging

from plato.callbacks import trainer as trainer_callbacks
from plato.config import Config


class PersonalizedModelCallback(trainer_callbacks.TrainerCallback):
    """A trainer callback to record the personalized model."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Saving the personalized model."""

        if trainer.is_final_personalization() or trainer.is_round_personalization():
            trainer.perform_personalized_model_checkpoint(config, **kwargs)
