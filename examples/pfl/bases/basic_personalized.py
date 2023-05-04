"""
The training and testing loops of PyTorch for personalized federated learning.

"""
import logging
import multiprocessing as mp
import os
import time
import warnings

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from plato.config import Config
from plato.trainers import basic
from plato.trainers import optimizers, lr_schedulers, loss_criterion, tracking
from plato.utils import checkpoint_operator
from plato.utils.filename_formatter import NameFormatter
from plato.utils import fonts

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic personalized federated learning trainer."""

    def get_checkpoint_dir_path(self):
        """Get the checkpoint path for current client."""
        checkpoint_path = Config.params["checkpoint_path"]
        return os.path.join(checkpoint_path, f"client_{self.client_id}")

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        loss_criterion_type = (
            Config().trainer.personalized_loss_criterion
            if hasattr(Config.trainer, "personalized_loss_criterion")
            else "CrossEntropyLoss"
        )
        loss_criterion_params = (
            Config().parameters.personalized_loss_criterion._asdict()
            if hasattr(Config.parameters, "personalized_loss_criterion")
            else {}
        )
        return loss_criterion.get(
            loss_criterion=loss_criterion_type,
            loss_criterion_params=loss_criterion_params,
        )

    def get_optimizer(self, model):
        """Returns the optimizer."""
        optimizer_name = Config().trainer.personalized_optimizer
        optimizer_params = Config().parameters.personalized_optimizer._asdict()

        return optimizers.get(
            model, optimizer_name=optimizer_name, optimizer_params=optimizer_params
        )

    def get_lr_scheduler(self, optimizer):
        """Returns the learning rate scheduler, if needed."""
        lr_scheduler = Config().trainer.personalized_lr_scheduler
        lr_params = Config().parameters.personalized_learning_rate._asdict()

        return lr_schedulers.get(
            optimizer,
            len(self.personalized_train_loader),
            lr_scheduler=lr_scheduler,
            lr_params=lr_params,
        )

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
        model_name=None,
        modelfile_prefix="personalized",
        rollback_round=None,
        location=None,
    ):
        """Rollback the model to be the previously one.
        By default, this functon rollbacks the personalized model.

        """
        rollback_round = (
            rollback_round if rollback_round is not None else self.current_round - 1
        )
        model_name = (
            model_name
            if model_name is not None
            else Config().trainer.personalized_model_name
        )
        location = location if location is not None else self.get_checkpoint_dir_path()
        modelfile_prefix = modelfile_prefix if modelfile_prefix is not None else None

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
        loaded_weights = rollback_status["model"]
        if modelfile_prefix == "personalized":
            self.personalized_model.load_state_dict(loaded_weights, strict=True)
        else:
            self.model.load_state_dict(loaded_weights, strict=True)

        logging.info(
            "[Client #%d] Rolled back the model from %s under %s.",
            self.client_id,
            filename,
            location,
        )
        # remove the weights for simplicity
        del rollback_status["model"]
        return rollback_status

    @staticmethod
    def save_personalized_accuracy(
        accuracy,
        current_round=None,
        epoch=None,
        accuracy_type="test_accuracy",
        filename=None,
        location=None,
    ):
        # pylint:disable=too-many-arguments
        """Saving the test accuracy to a file."""
        to_save_dir, filename = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".csv",
        )
        to_save_path = os.path.join(to_save_dir, filename)
        current_round = current_round if current_round is not None else 0
        current_epoch = epoch if epoch is not None else 0
        acc_dataframe = pd.DataFrame(
            {"round": current_round, "epoch": current_epoch, accuracy_type: accuracy},
            index=[0],
        )

        is_use_header = not os.path.exists(to_save_path)
        acc_dataframe.to_csv(to_save_path, index=False, mode="a", header=is_use_header)

    @staticmethod
    def load_personalized_accuracy(
        current_round=None,
        accuracy_type="test_accuracy",
        filename=None,
        location=None,
    ):
        """Loading the test accuracy from a file."""
        to_save_dir, filename = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".acc",
        )
        load_path = os.path.join(to_save_dir, filename)
        loaded_rounds_accuracy = pd.read_csv(load_path)
        if current_round is None:
            # default use the last row
            desired_row = loaded_rounds_accuracy.iloc[-1]
        else:
            desired_row = loaded_rounds_accuracy.loc[
                loaded_rounds_accuracy["round"] == current_round
            ]
            desired_row = loaded_rounds_accuracy.iloc[-1]

        accuracy = desired_row[accuracy_type]

        return accuracy

    def checkpoint_personalized_accuracy(self, accuracy, current_round, epoch, run_id):
        """Save the personaliation accuracy to the results dir."""
        result_path = Config().params["result_path"]

        save_location = os.path.join(result_path, "client_" + str(self.client_id))

        filename = NameFormatter.get_format_name(
            client_id=self.client_id, suffix="personalized_accuracy", ext="csv"
        )
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_accuracy(
            accuracy,
            current_round=current_round,
            epoch=epoch,
            accuracy_type="personalized_accuracy",
            filename=filename,
            location=save_location,
        )
