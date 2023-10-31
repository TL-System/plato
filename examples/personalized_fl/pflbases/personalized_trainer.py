"""
A trainer to support the personalized federated learning
in the final round.
"""
import logging
import os

from plato.config import Config
from plato.trainers import basic
from plato.trainers import optimizers, lr_schedulers, loss_criterion


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

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
            return super().get_optimizer(self.model)

        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()

        return optimizers.get(
            self.model,
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
