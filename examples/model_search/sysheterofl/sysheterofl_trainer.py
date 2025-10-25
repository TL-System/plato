"""
The trainer for system-heterogenous federated learning through architecture search.

This trainer uses the composable trainer architecture with a custom callback to reinitialize the
model after parent initialization.

Reference: D. Yao, "Exploring System-Heterogeneous Federated Learning with Dynamic Model Selection,"
https://arxiv.org/abs/2409.08858.
"""

import logging

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import basic


class ModelReinitializationCallback(TrainerCallback):
    """
    Callback to reinitialize the model after trainer initialization.

    This is needed for SysHeteroFL to create the model with specific parameters.
    """

    def __init__(self, model_class):
        """Initialize with model class."""
        self.model_class = model_class

    def on_trainer_initialized(self, trainer, **kwargs):
        """Reinitialize the model with config parameters."""
        if self.model_class is not None:
            trainer.model = self.model_class(**Config().parameters.model._asdict())
            logging.info("Model reinitialized with config parameters for SysHeteroFL")


class ServerTrainer(basic.Trainer):
    """
    A federated learning trainer of SysHeteroFL, used by the server.

    This trainer reinitializes the model with specific parameters from config
    and stores both the model class and the biggest network configuration.
    It uses the composable trainer architecture with a callback for model
    reinitialization.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the SysHeteroFL server trainer.

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

        # Store model class and reinitialize model with config parameters
        # Note: This maintains backward compatibility with the old behavior
        self.model_class = model
        if model is not None:
            self.model = model(**Config().parameters.model._asdict())
        self.biggest_net_config = None

    def test(self, testset, sampler=None, **kwargs):  # pylint: disable=unused-argument
        """Run server-side evaluation without spawning subprocesses."""
        logging.info(
            "[SysHeteroFL] Running in-process evaluation to avoid shared memory limits."
        )
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]
        return self.test_model(config, testset, sampler=sampler, **kwargs)
