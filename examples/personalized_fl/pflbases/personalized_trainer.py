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
from plato.models import registry as models_registry
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter

from pflbases import trainer_utils
from pflbases import checkpoint_operator

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        self.personalized_model = None

        # personalized model name and the file prefix
        # used to save the model
        self.personalized_model_name = (
            Config().algorithm.personalization.model_name
            if hasattr(Config().algorithm.personalization, "model_name")
            else Config().trainer.model_name
        )
        self.personalized_model_checkpoint_prefix = "personalized"

    def define_personalized_model(self, personalized_model_cls):
        """Define the personalized model to this trainer."""
        trainer_utils.set_random_seeds(self.client_id)

        if personalized_model_cls is None:
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
            self.personalized_model = personalized_model_cls.get()

        logging.info(
            "[Client #%d] Defined the personalized model: %s",
            self.client_id,
            self.personalized_model_name,
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
            Config().parameters.personalization.loss_criterion._asdict()
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
        if not self.is_final_personalization():
            return super().get_optimizer(model)

        logging.info("[Client #%d] Using the personalized optimizer.", self.client_id)

        return self.get_personalized_optimizer()

    def get_lr_scheduler(self, config, optimizer):
        """Returns the learning rate scheduler, if needed."""
        if not self.is_final_personalization():
            return super().get_lr_scheduler(config, optimizer)

        logging.info(
            "[Client #%d] Using the personalized lr_scheduler.", self.client_id
        )

        return self.get_personalized_lr_scheduler(config, optimizer)

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        if not self.is_final_personalization():
            return super().get_loss_criterion()

        logging.info(
            "[Client #%d] Using the personalized loss_criterion.", self.client_id
        )

        return self.get_personalized_loss_criterion()

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader for personalization."""
        if self.is_final_personalization() and hasattr(
            Config().algorithm, "personalization"
        ):
            personalized_config = Config().algorithm.personalization._asdict()
            if "batch_size" in personalized_config:
                batch_size = personalized_config["batch_size"]

        return super().get_train_loader(batch_size, trainset, sampler, **kwargs)

    def copy_model_to_personalized_model(self, config):
        """Copying the model to the personalized model."""
        self.personalized_model.load_state_dict(self.model.state_dict(), strict=True)
        logging.info(
            fonts.colourize(
                "[Client #%d] copied the model [%s] to personalized model [%s].",
                colour="blue",
            ),
            self.client_id,
            Config().trainer.model_name,
            Config().algorithm.personalization.model_name,
        )

    def preprocess_models(self, config):
        """Before running, we need to process the model and the personalized model.

        This function is required to be revised based on the specific condition of the
        personalized FL algorithm.
        """
        # By default:
        # before performing the final personalization to optimized the personalized model,
        # each client has to copy the global model to the personalized model by default.

        if self.is_final_personalization():
            self.copy_model_to_personalized_model(config)

    def train_run_start(self, config):
        """Before running, convert the config to be ones for personalization."""

        self.preprocess_models(config)

        if self.is_final_personalization():
            personalized_config = Config().algorithm.personalization._asdict()
            config.update(personalized_config)
            # the model name is needed to be maintained here
            # as Plato will use config["model_name"] to save the model
            # and then load the saved model relying on
            # Config().trainer.model_name
            config["model_name"] = Config().trainer.model_name

            self.personalized_model.to(self.device)
            self.personalized_model.train()

    def postprocess_models(self, config):
        """Before running, process the personalized model."""

        # pflbases will always copy the trained model to the personalized model as
        # the local update performed on the received global model is the core of
        # federated learning and should be Mandatory.
        # Therefore, pflbases assumes that the personalized model is optimized
        # as one part of the local update.
        # By default:
        # the updated global model will be copied to the personalized model
        if self.is_round_personalization() and not self.is_final_personalization():
            self.copy_model_to_personalized_model(config)

    def train_run_end(self, config):
        """Copy the trained model to the untrained one."""
        super().train_run_end(config)

        self.postprocess_models(config)

        if self.is_round_personalization() or self.is_final_personalization():
            self.perform_personalized_model_checkpoint(config)

    def model_forward(self, examples):
        """Forward the input examples to the model."""

        return self.model(examples)

    def personalized_model_forward(self, examples, **kwargs):
        """Forward the input examples to the personalized model."""

        return self.personalized_model(examples)

    def forward_examples(self, examples, **kwargs):
        """Forward the examples through one model."""

        if self.is_final_personalization():
            return self.personalized_model_forward(examples, **kwargs)
        else:
            return self.model_forward(examples, **kwargs)

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

    def test_personalized_model(self, config, testset, sampler=None, **kwargs):
        """Test the personalized model."""
        # Define the test phase of the eval stage

        logging.info("[Client #%d] Testing the personalized model.", self.client_id)

        self.personalized_model.eval()
        self.personalized_model.to(self.device)

        batch_size = config["batch_size"]
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for _, (examples, labels) in enumerate(test_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.personalized_model_forward(examples, split="test")
                outputs = self.process_personalized_outputs(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        self.personalized_model.train()

        return accuracy

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Testing the model to report the accuracy of the local model or the
        personalized model."""

        if self.is_round_personalization() or self.is_final_personalization():
            return self.test_personalized_model(config, testset, sampler=None, **kwargs)
        else:
            return super().test_model(config, testset, sampler, **kwargs)

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
                "[Client #%d] Loaded a Personalized model from %s under %s.",
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

    def is_final_personalization(self):
        """Get whether the client is performing the final personalization.
        whether the client is performing the final personalization
        the final personalization is mandatory
        """
        if self.current_round > Config().trainer.rounds:
            return True
        return False

    def is_round_personalization(self):
        """Get whether the client is performing the round personalization.
        whether the client is perfomring the personalization in the current round
        this round personalization should be determined by the user
        depending on the algorithm.
        """

        if (
            hasattr(Config().algorithm.personalization, "do_personalization_per_round")
            and Config().algorithm.personalization.do_personalization_per_round
        ):
            return True
        return False
