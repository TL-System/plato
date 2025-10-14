"""
AnyCostFL algorithm trainer.

This trainer uses the composable trainer architecture with a custom callback
to reinitialize the model after parent initialization.
"""

import logging

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.basic import Trainer


class ModelReinitializationCallback(TrainerCallback):
    """
    Callback to reinitialize the model after trainer initialization.

    This is needed for AnyCostFL to create the model with specific parameters.
    """

    def __init__(self, model_class):
        """Initialize with model class."""
        self.model_class = model_class

    def on_trainer_initialized(self, trainer, **kwargs):
        """Reinitialize the model with config parameters."""
        if self.model_class is not None:
            trainer.model = self.model_class(**Config().parameters.model._asdict())
            logging.info("Model reinitialized with config parameters for AnyCostFL")


class ServerTrainer(Trainer):
    """
    A federated learning trainer of AnyCostFL, used by the server.

    This trainer reinitializes the model with specific parameters from config.
    It uses the composable trainer architecture with a callback for model
    reinitialization.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the AnyCostFL server trainer.

        Arguments:
            model: The model class or instance
            callbacks: List of callback classes or instances
        """
        # Create model reinitialization callback
        reinit_callback = ModelReinitializationCallback(model)

        # Combine with provided callbacks
        all_callbacks = [reinit_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent trainer
        super().__init__(model=model, callbacks=all_callbacks)

        # Reinitialize model with config parameters
        # Note: This maintains backward compatibility with the old behavior
        if model is not None:
            self.model = model(**Config().parameters.model._asdict())
