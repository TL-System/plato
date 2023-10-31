"""
A trainer to support the personalized federated learning.

"""
import logging
import os
import warnings

from plato.config import Config
from plato.trainers import basic
from plato.trainers import optimizers, lr_schedulers, loss_criterion
from pflbases.filename_formatter import NameFormatter

from pflbases import trainer_utils

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        # The model name and the file prefix
        # used to save the model.
        self.model_name = Config().trainer.model_name
        self.local_model_prefix = "local"

    def reinitialize_local_model(self):
        """Reinitialize the local model based on the client id
        as the random seed to ensure that each client id corresponds to
        the specific model."""
        trainer_utils.set_random_seeds(self.client_id)
        self.model.apply(trainer_utils.weights_reinitialize)
        logging.info(
            "[Client #%d] Re-initialized the local model with the random seed %d.",
            self.client_id,
            self.client_id,
        )

    def get_personalized_model_params(self):
        """Get the params of the personalized model."""
        if hasattr(Config().parameters, "personalization"):
            return Config().parameters.personalization.model._asdict()
        else:
            if hasattr(Config().parameters, "model"):
                return Config().parameters.model._asdict()
            else:
                return {}

    def get_checkpoint_dir_path(self):
        """Get the checkpoint path for current client."""
        checkpoint_path = Config.params["checkpoint_path"]
        return os.path.join(checkpoint_path, f"client_{self.client_id}")

    def get_personalized_loss_criterion(self):
        """Getting the loss_criterion for personalized model."""

        if not hasattr(Config().algorithm, "personalization") or not hasattr(
            Config().algorithm.personalization, "loss_criterion"
        ):
            return super().get_loss_criterion()

        loss_criterion_type = Config().algorithm.personalization.loss_criterion

        loss_criterion_params = (
            {}
            if not hasattr(Config().parameters.personalization, "loss_criterion")
            else Config().parameters.personalization.loss_criterion._asdict()
        )

        return loss_criterion.get(
            loss_criterion=loss_criterion_type,
            loss_criterion_params=loss_criterion_params,
        )

    def get_personalized_optimizer(self):
        """Getting the optimizer for personalized model."""

        if not hasattr(Config().algorithm, "personalization") or not hasattr(
            Config().algorithm.personalization, "optimizer"
        ):
            return super().get_optimizer(self.personalized_model)

        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()

        return optimizers.get(
            self.personalized_model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def get_personalized_lr_scheduler(self, config, optimizer):
        """Getting the lr scheduler for personalized model."""

        if not hasattr(Config().algorithm, "personalization") or not hasattr(
            Config().parameters, "personalization"
        ):
            return super().get_lr_scheduler(config, optimizer)

        lr_scheduler = Config().algorithm.personalization.lr_scheduler
        lr_params = Config().parameters.personalization.learning_rate._asdict()

        return lr_schedulers.get(
            optimizer,
            len(self.train_loader),
            lr_scheduler=lr_scheduler,
            lr_params=lr_params,
        )

    def get_optimizer(self, model):
        """Returns the optimizer."""
        if not self.do_final_personalization:
            return super().get_optimizer(model)

        logging.info("[Client #%d] Using the personalized optimizer.", self.client_id)

        return self.get_personalized_optimizer()

    def get_lr_scheduler(self, config, optimizer):
        """Returns the learning rate scheduler, if needed."""
        if not self.do_final_personalization:
            return super().get_lr_scheduler(config, optimizer)

        logging.info(
            "[Client #%d] Using the personalized lr_scheduler.", self.client_id
        )

        return self.get_personalized_lr_scheduler(config, optimizer)

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        if not self.do_final_personalization:
            return super().get_loss_criterion()

        logging.info(
            "[Client #%d] Using the personalized loss_criterion.", self.client_id
        )

        return self.get_personalized_loss_criterion()

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader for personalization."""
        if self.do_final_personalization and hasattr(
            Config().algorithm, "personalization"
        ):
            personalized_config = Config().algorithm.personalization._asdict()
            if "batch_size" in personalized_config:
                batch_size = personalized_config["batch_size"]

        return super().get_train_loader(batch_size, trainset, sampler, **kwargs)

    def train_run_start(self, config):
        """Before running, convert the config to be ones for personalization."""

        if self.do_final_personalization:
            personalized_config = Config().algorithm.personalization._asdict()
            config.update(personalized_config)
            # the model name is needed to be maintained here
            # as Plato will use config["model_name"] to save the model
            # and then load the saved model relying on
            # Config().trainer.model_name
            config["model_name"] = Config().trainer.model_name

    def train_run_end(self, config):
        """Copy the trained model to the untrained one."""
        super().train_run_end(config)

        self.perform_local_model_checkpoint(config=config)

    def get_model_checkpoint_path(
        self, model_name: str, prefix=None, round_n=None, epoch_n=None
    ):
        """Getting the path of the personalized model."""
        current_round = self.current_round if round_n is None else round_n

        save_location = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name=model_name,
            round_n=current_round,
            epoch_n=epoch_n,
            prefix=prefix,
            ext="pth",
        )

        return save_location, filename

    def perform_local_model_checkpoint(self, **kwargs):
        """Performing the saving for the personalized model with
        necessary learning parameters."""
        round_n = kwargs.pop("round") if "round" in kwargs else self.current_round
        epoch_n = kwargs.pop("epoch") if "epoch" in kwargs else None
        model_name = self.model_name
        prefix = self.local_model_prefix
        save_location, filename = self.get_model_checkpoint_path(
            model_name=model_name,
            prefix=prefix,
            round_n=round_n,
            epoch_n=epoch_n,
        )

        self.save_model(filename=filename, location=save_location)

        # Always remove the expired checkpoints.
        self.remove_expired_checkpoints(
            model_name=model_name, prefix=prefix, round_n=round_n
        )

    def remove_expired_checkpoints(self, model_name, prefix, **kwargs):
        """Removing invalid checkpoints under the checkpoints_dir.
        This function will only maintain the initial one and latest one.
        """
        current_round = (
            self.current_round if "round_n" not in kwargs else kwargs["round_n"]
        )
        for round_id in range(1, current_round):
            save_location, filename = self.get_model_checkpoint_path(
                model_name=model_name,
                prefix=prefix,
                round_n=round_id,
            )
            if os.path.exists(os.path.join(save_location, filename)):
                os.remove(os.path.join(save_location, filename))
