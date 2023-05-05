"""
The training and testing loops of PyTorch for personalized federated learning.

"""
import logging
import os
import warnings

import pandas as pd
import torch

from plato.config import Config
from plato.trainers import basic
from plato.trainers import optimizers, lr_schedulers, loss_criterion
from plato.utils import checkpoint_operator
from plato.models import registry as models_registry

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        self.personalized_model = None

        # obtain the personalized model name
        self.personalized_model_name = Config().algorithm.personalization.model_name
        self.personalized_model_checkpoint_prefix = "personalized"

        # training mode
        # the trainer should either perform the normal training
        # or the personalized training
        self.personalized_training = False

    def set_training_mode(self, personalized_mode: bool):
        """Set the learning model of this trainer.

        The learning mode must be set by the client.
        """
        self.personalized_training = personalized_mode

    def define_personalized_model(self, personalized_model):
        """Define the personalized model to this trainer."""
        if personalized_model is None:

            pers_model_type = (
                Config().algorithm.personalization.model_type
                if hasattr(Config().algorithm.personalization, "model_type")
                else self.personalized_model_name.split("_")[0]
            )
            pers_model_params = self.get_personalized_model_params()
            self.personalized_model = models_registry.get(
                model_name=self.personalized_model_name,
                model_type=pers_model_type,
                model_params=pers_model_params,
            )
        else:
            self.personalized_model = personalized_model()

        logging.info(
            "[Client #%d] defined the personalized model: %s",
            self.client_id,
            self.personalized_model_name,
        )

    def get_personalized_model_params(self):
        """Get the params of the personalized model."""
        return Config().parameters.personalization.model._asdict()

    def get_checkpoint_dir_path(self):
        """Get the checkpoint path for current client."""
        checkpoint_path = Config.params["checkpoint_path"]
        return os.path.join(checkpoint_path, f"client_{self.client_id}")

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        if not self.personalized_training:
            return super().get_loss_criterion()

        loss_criterion_type = Config().algorithm.personalization.loss_criterion
        loss_criterion_params = (
            Config().parameters.personalization.loss_criterion._asdict()
        )
        return loss_criterion.get(
            loss_criterion=loss_criterion_type,
            loss_criterion_params=loss_criterion_params,
        )

    def get_optimizer(self, model):
        """Returns the optimizer."""
        if not self.personalized_training:
            return super().get_optimizer(model)

        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()

        return optimizers.get(
            self.personalized_model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def get_lr_scheduler(self, optimizer):
        """Returns the learning rate scheduler, if needed."""
        if not self.personalized_training:
            return super().get_lr_scheduler(optimizer)

        lr_scheduler = Config().algorithm.personalization.lr_scheduler
        lr_params = Config().parameters.personalization.learning_rate._asdict()

        return lr_schedulers.get(
            optimizer,
            len(self.train_loader),
            lr_scheduler=lr_scheduler,
            lr_params=lr_params,
        )

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the batch size of personalization."""
        batch_size = batch_size
        if self.personalized_training:
            personalized_config = Config().algorithm.personalization._asdict()
            batch_size = personalized_config["batch_size"]

        return super().get_train_loader(batch_size, trainset, sampler, **kwargs)

    def train_run_start(self, config):
        """Before running, convert the config to be ones for personalization."""
        if self.personalized_training:
            personalized_config = Config().algorithm.personalization._asdict()
            config["batch_size"] = personalized_config["batch_size"]
            config["epochs"] = personalized_config["epochs"]

            self.personalized_model.to(self.device)
            self.personalized_model.train()

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        self.optimizer.zero_grad()

        if not self.personalized_training:
            outputs = self.model(examples)
        else:
            outputs = self.personalized_model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss

    @staticmethod
    @torch.no_grad()
    def reset_weight(module: torch.nn.Module):
        """
        refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html

        One model can be reset by
        # Applying fn recursively to every submodule see:
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        model.apply(fn=weight_reset)
        """

        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    def rollback_model(
        self,
        rollback_round=None,
        model_name=None,
        modelfile_prefix=None,
        location=None,
    ):
        """Rollback the model to be the previously one.
        By default, this functon rollbacks the personalized model.

        """
        rollback_round = (
            rollback_round if rollback_round is not None else self.current_round - 1
        )
        model_name = (
            model_name if model_name is not None else self.personalized_model_name
        )
        modelfile_prefix = (
            modelfile_prefix
            if modelfile_prefix is not None
            else self.personalized_model_checkpoint_prefix
        )
        location = location if location is not None else self.get_checkpoint_dir_path()

        filename, ckpt_oper = checkpoint_operator.load_client_checkpoint(
            client_id=self.client_id,
            checkpoints_dir=location,
            model_name=model_name,
            current_round=rollback_round,
            run_id=None,
            epoch=None,
            prefix=modelfile_prefix,
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )

        rollback_status = ckpt_oper.load_checkpoint(checkpoint_name=filename)
        logging.info(
            "[Client #%d] Rolled back the model from %s under %s.",
            self.client_id,
            filename,
            location,
        )
        return rollback_status

    def create_unique_personalized_model(self, filename):
        """Reset the model parameters."""
        checkpoint_dir_path = self.get_checkpoint_dir_path()
        # reset the model for this client
        # thus, different clients have different init parameters
        self.personalized_model.apply(self.reset_weight)
        self.save_model(
            filename=filename,
            location=checkpoint_dir_path,
        )
        logging.info(
            "[Client #%d] Created the unique personalized model as %s and saved to %s.",
            self.client_id,
            filename,
            checkpoint_dir_path,
        )
