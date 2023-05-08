"""
The training and testing loops for personalized federated learning.

"""
import logging
import os
import warnings

import torch

from plato.config import Config
from plato.trainers import basic
from plato.trainers import optimizers, lr_schedulers, loss_criterion
from plato.utils import checkpoint_operator
from plato.models import registry as models_registry
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter

from bases.trainer_callbacks.base_callbacks import (
    PersonalizedLogProgressCallback,
)

from bases.trainer_utils import checkpoint_personalized_accuracy

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        # clear the original callbacks but only hold the
        # desired ones
        self.callback_handler.clear_callbacks()
        self.callbacks = [
            PersonalizedLogProgressCallback,
        ]
        if callbacks is not None:
            self.callbacks.extend(callbacks)

        # only add the customized callbacks
        self.callback_handler.add_callbacks(self.callbacks)

        self.personalized_model = None

        # obtain the personalized model name
        self.personalized_model_name = Config().algorithm.personalization.model_name
        self.personalized_model_checkpoint_prefix = "personalized"

        # training mode
        # the trainer should either perform the normal training
        # or the personalized training
        self.personalized_learning = False

        # personalized model evaluation
        self.testset = None
        self.testset_sampler = None

    def set_training_mode(self, personalized_mode: bool):
        """Set the learning model of this trainer.

        The learning mode must be set by the client.
        """
        self.personalized_learning = personalized_mode

    def set_testset(self, dataset):
        """set the testset."""
        self.testset = dataset

    def set_testset_sampler(self, sampler):
        """set the sampler for the testset."""
        self.testset_sampler = sampler

    # pylint: disable=unused-argument
    def get_test_loader(self, batch_size, testset, sampler, **kwargs):
        """
        Creates an instance of the testloader.

        Arguments:
        batch_size: the batch size.
        testset: the training dataset.
        sampler: the sampler for the testloader to use.
        """
        return torch.utils.data.DataLoader(
            dataset=testset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

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
        if not self.personalized_learning:
            return super().get_loss_criterion()

        loss_criterion_type = Config().algorithm.personalization.loss_criterion
        loss_criterion_params = (
            Config().parameters.personalization.loss_criterion._asdict()
        )
        return loss_criterion.get(
            loss_criterion=loss_criterion_type,
            loss_criterion_params=loss_criterion_params,
        )

    def get_personalized_optimizer(self):
        """Getting the optimizer for personalized model."""
        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()

        return optimizers.get(
            self.personalized_model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def get_optimizer(self, model):
        """Returns the optimizer."""
        if not self.personalized_learning:
            return super().get_optimizer(model)

        return self.get_personalized_optimizer()

    def get_lr_scheduler(self, config, optimizer):
        """Returns the learning rate scheduler, if needed."""
        if not self.personalized_learning:
            return super().get_lr_scheduler(config, optimizer)

        lr_scheduler = Config().algorithm.personalization.lr_scheduler
        lr_params = Config().parameters.personalization.learning_rate._asdict()

        return lr_schedulers.get(
            optimizer,
            len(self.train_loader),
            lr_scheduler=lr_scheduler,
            lr_params=lr_params,
        )

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader for personalization."""
        if self.personalized_learning:
            personalized_config = Config().algorithm.personalization._asdict()
            batch_size = personalized_config["batch_size"]

        return super().get_train_loader(batch_size, trainset, sampler, **kwargs)

    def preprocess_personalized_model(self, config):
        """Before running, process the personalized model."""
        logging.info(
            fonts.colourize(
                "[Client #%d] assings the model [%s] to personalized model [%s].",
                colour="blue",
            ),
            self.client_id,
            Config().trainer.model_name,
            Config().algorithm.personalization.model_name,
        )

        # load the received model to be personalized model
        self.personalized_model.load_state_dict(self.model.state_dict(), strict=True)

    def train_run_start(self, config):
        """Before running, convert the config to be ones for personalization."""
        self.preprocess_personalized_model(config)
        if self.personalized_learning:
            personalized_config = Config().algorithm.personalization._asdict()
            config.update(personalized_config)

            self.personalized_model.to(self.device)
            self.personalized_model.train()

    def model_forward(self, examples):
        """Forward the input examples to the model."""

        return self.model(examples)

    def personalized_model_forward(self, examples):
        """Forward the input examples to the personalized model."""

        return self.personalized_model(examples)

    def forward_examples(self, examples):
        """Forward the examples through one model."""

        if self.personalized_learning:
            return self.personalized_model_forward(examples)
        else:
            return self.model_forward(examples)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        outputs = self.forward_examples(examples)

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
        """Reset model parameters."""
        checkpoint_dir_path = self.get_checkpoint_dir_path()
        # reset the model for this client
        # thus, different clients have different init parameters
        self.personalized_model.apply(self.reset_weight)

        logging.info(
            fonts.colourize(
                "[Client #%d] Created the unique personalized model as %s and saved to %s.",
                colour="blue",
            ),
            self.client_id,
            filename,
            checkpoint_dir_path,
        )

        self.save_personalized_model(
            filename=filename,
            location=checkpoint_dir_path,
        )

    def test_personalized_model(self, config, **kwargs):
        """Test the personalized model."""
        # Define the test phase of the eval stage

        self.personalized_model.eval()
        self.personalized_model.to(self.device)

        data_loader = self.get_test_loader(
            config["batch_size"],
            testset=self.testset,
            sampler=self.testset_sampler.get(),
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for _, (examples, labels) in enumerate(data_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.personalized_model_forward(examples)

                outputs = self.process_personalized_outputs(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        outputs = {"accuracy": accuracy}

        self.personalized_model.train()

        return outputs

    def perform_personalized_metric_checkpoint(self, config):
        """Performing the test for the personalized model and saving the accuracy to
        checkpoint file."""
        result_path = Config().params["result_path"]
        test_outputs = self.test_personalized_model(config)

        checkpoint_personalized_accuracy(
            result_path,
            client_id=self.client_id,
            accuracy=test_outputs["accuracy"],
            current_round=self.current_round,
            epoch=self.current_epoch,
            run_id=None,
        )

    def perform_personalized_model_checkpoint(self, config, epoch=None, **kwargs):
        """Performing the saving for the personalized model with
        necessary learning parameters."""
        current_round = self.current_round

        personalized_model_name = self.personalized_model_name
        save_location = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name=personalized_model_name,
            round_n=current_round,
            epoch_n=epoch,
            run_id=None,
            prefix=self.personalized_model_checkpoint_prefix,
            ext="pth",
        )
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_model(
            filename=filename, location=save_location, **kwargs
        )

    def save_personalized_model(self, filename=None, location=None, **kwargs):
        """Saving the model to a file."""

        ckpt_oper = checkpoint_operator.CheckpointsOperator(checkpoints_dir=location)
        ckpt_oper.save_checkpoint(
            model_state_dict=self.personalized_model.state_dict(),
            checkpoints_name=[filename],
            **kwargs,
        )

        logging.info(
            fonts.colourize(
                "[Client #%d] Saved personalized model to %s under %s.", colour="blue"
            ),
            self.client_id,
            filename,
            location,
        )

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""

        ckpt_oper = checkpoint_operator.CheckpointsOperator(checkpoints_dir=location)
        self.personalized_model.load_state_dict(
            ckpt_oper.load_checkpoint(filename)["model"], strict=True
        )

        logging.info(
            fonts.colourize(
                "[Client #%d] Loading a Personalized model from %s under %s.",
                colour="blue",
            ),
            self.client_id,
            filename,
            location,
        )

    @staticmethod
    def process_personalized_outputs(outputs):
        """
        Method called to process outputs of the personalized model.
        """
        return outputs
