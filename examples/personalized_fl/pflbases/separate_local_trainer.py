"""
A trainer to support a separate client's local model and the global model.
"""
import logging
import warnings

from plato.config import Config
from plato.models import registry as models_registry

from pflbases import trainer_utils
from pflbases import personalized_trainer

warnings.simplefilter("ignore")


class Trainer(personalized_trainer.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        # get the model name
        self.model_name = Config().trainer.model_name
        self.local_model_prefix = "local"

    def define_local_model(self, custom_model):
        """Define the local model to this trainer."""
        trainer_utils.set_random_seeds(self.client_id)
        if custom_model is None:
            self.model = models_registry.get()
        else:
            self.model = custom_model.get()

        logging.info(
            "[Client #%d] Defined the local model: %s",
            self.client_id,
            self.model_name,
        )

    def train_run_end(self, config):
        """Copy the trained model to the untrained one."""
        super().train_run_end(config)

        self.perform_local_model_checkpoint(config)

    def perform_local_model_checkpoint(self, config, **kwargs):
        """Performing the saving for the personalized model with
        necessary learning parameters."""
        round_n = kwargs.pop("round") if "round" in kwargs else self.current_round
        epoch_n = kwargs.pop("epoch") if "epoch" in kwargs else None
        save_location, filename = self.get_model_checkpoint_path(
            model_name=self.model_name,
            prefix=self.local_model_prefix,
            round_n=round_n,
            epoch_n=epoch_n,
        )

        self.save_model(filename=filename, location=save_location)
