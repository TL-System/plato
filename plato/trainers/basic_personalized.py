"""
The training and testing loops of PyTorch for personalized federated learning.

Very important notes:
    1.- For all model saving filename that contains 'epoch', it is a checkpoint file.
    Our checkpoint file is saved by the function 'perform_client_checkpoint_saving'.
    Therefore, loading the model parameter of the checkpoint file should be:
        load_checkpoint(filepath)['model']

    2.- For all model saving filename without containing 'epoch', there are two
    types of model saving files, including:
        - lenet5__client4_round1.pth -> corresponds to self.model
        - personalized_lenet5__client4_round1.pth -> corresponds to self.personalized_model

    For the self.model, we save it as the checkpoint as this model can
    be regarded as the trained model during the whole training stage of FL.
        -> Therefore, loading the model parameter of the checkpoint file should be:
        load_checkpoint(filepath)['model']

    These settings induce our recall self.model and self.personalized_model in each round
    Please access the 'load_personalized_model' and 'load_payload' in the
    clients/pers_simple.py.

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
from plato.utils.filename_formatter import get_format_name
from plato.utils import fonts

warnings.simplefilter("ignore")


class Trainer(basic.Trainer):
    # pylint:disable=too-many-public-methods
    """A basic federated learning trainer, used by both the client and the server."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the client's personalized model
        # to perform the evaluation stage of the ssl methods
        # the client must assign its own personalized model
        #  to its trainer
        self.personalized_model = None
        self.personalized_model_state_dict = None

        self._loss_tracker = tracking.LossTracker()

    def set_client_personalized_model(self, personalized_model):
        """Setting the client's personalized model"""
        self.personalized_model = personalized_model

    def get_checkpoint_dir_path(self):
        """Get the checkpoint path for current client."""
        checkpoint_path = Config.params["checkpoint_path"]
        return os.path.join(checkpoint_path, f"client_{self.client_id}")

    def get_personalized_loss_criterion(self):
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

    def get_personalized_optimizer(self, model):
        """Returns the optimizer."""
        optimizer_name = Config().trainer.personalized_optimizer
        optimizer_params = Config().parameters.personalized_optimizer._asdict()

        return optimizers.get(
            model, optimizer_name=optimizer_name, optimizer_params=optimizer_params
        )

    def get_personalized_lr_scheduler(self, optimizer):
        """Returns the learning rate scheduler, if needed."""
        lr_scheduler = Config().trainer.personalized_lr_scheduler
        lr_params = Config().parameters.personalized_learning_rate._asdict()

        return lr_schedulers.get(
            optimizer,
            len(self.train_loader),
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

    @staticmethod
    def process_save_path(filename, location, work_model_name, desired_extenstion):
        """Process the input arguments to obtain the final saving path."""
        # default saving everything to the model path
        location = Config().params["model_path"] if location is None else location
        # set the model_type to
        #  - "model_name" to obtain the global model's name
        #  - "personalized_model_name" to obtain the personalized model's name
        #  - or any other model you defined and want to process
        model_name = getattr(Config().trainer, work_model_name)
        try:
            if not os.path.exists(location):
                os.makedirs(location)
        except FileExistsError:
            pass

        filename = model_name if filename is None else filename

        # check the file extension
        save_prefix, save_extension = os.path.splitext(filename)
        if save_extension != desired_extenstion:
            filename = save_prefix + desired_extenstion

        return location, filename

    def save_personalized_model(self, filename=None, location=None):
        """Saving the model to a file."""
        # process the arguments to obtain the to save path
        to_save_dir, filename = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth",
        )
        ckpt_oper = checkpoint_operator.CheckpointsOperator(checkpoints_dir=to_save_dir)
        ckpt_oper.save_checkpoint(
            model_state_dict=self.personalized_model.state_dict(),
            checkpoints_name=[filename],
        )

        logging.info(
            "[Client #%d] Personalized Model saved to %s under %s.",
            self.client_id,
            filename,
            to_save_dir,
        )

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        # process the arguments to obtain the to save path
        to_load_dir, filename = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth",
        )
        ckpt_oper = checkpoint_operator.CheckpointsOperator(checkpoints_dir=to_load_dir)
        self.personalized_model.load_state_dict(
            ckpt_oper.load_checkpoint(filename)["model"], strict=True
        )

        logging.info(
            "[Client #%d] Loading a Personalized model from %s under %s.",
            self.client_id,
            filename,
            to_load_dir,
        )

    def rollback_model(
        self,
        model_name=None,
        modelfile_prefix=None,
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
        loaded_weights = ckpt_oper.load_checkpoint(checkpoint_name=filename)["model"]
        if modelfile_prefix == "personalized":
            self.trainer.personalized_model.load_state_dict(loaded_weights, strict=True)
        else:
            self.trainer.model.load_state_dict(loaded_weights, strict=True)

        logging.info(
            "[Client #%d] Rollbacking a model from %s under %s.",
            self.client_id,
            filename,
            location,
        )

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

        filename = get_format_name(
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

    def save_encoded_data(
        self, encoded_data, data_labels, filename=None, location=None
    ):
        """Save the encoded data (np.narray)."""
        # convert the list to tensor
        encoded_data = torch.cat(encoded_data, axis=0)
        data_labels = torch.cat(data_labels)
        # combine the label to the final of the 2d
        to_save_data = torch.cat([encoded_data, data_labels.reshape(-1, 1)], dim=1)
        to_save_narray_data = to_save_data.detach().to("cpu").numpy()

        # process the arguments to obtain the to save path
        to_save_dir, filename = self.process_save_path(
            filename, location, work_model_name="model_name", desired_extenstion=".npy"
        )
        to_save_path = os.path.join(to_save_dir, filename)
        np.save(to_save_path, to_save_narray_data, allow_pickle=True)
        logging.info(
            "[Client #%d] Saving encoded data to %s.", self.client_id, to_save_path
        )

    def checkpoint_encoded_samples(
        self,
        encoded_samples,
        encoded_labels,
        current_round,
        epoch,
        run_id,
        encoded_type="trainEncoded",
    ):
        # pylint:disable=too-many-arguments
        """Save the encoded data to the results dir."""

        result_path = Config().params["result_path"]
        save_location = os.path.join(result_path, "client_" + str(self.client_id))
        save_filename = get_format_name(
            client_id=self.client_id,
            round_n=current_round,
            epoch_n=epoch,
            run_id=run_id,
            suffix=encoded_type,
            ext="npy",
        )

        self.save_encoded_data(
            encoded_data=encoded_samples,
            data_labels=encoded_labels,
            filename=save_filename,
            location=save_location,
        )

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):
        """The operation of performing the evaluation on the testset with the defined model."""
        # Define the test phase of the eval stage
        acc_meter = tracking.LossTracker()
        defined_model.eval()
        defined_model.to(self.device)

        correct = 0
        encoded_samples = []
        loaded_labels = []

        acc_meter.reset()
        for _, (examples, labels) in enumerate(to_eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                preds = defined_model(examples).argmax(dim=1)
                correct = (preds == labels).sum()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                encoded_samples.append(examples)
                loaded_labels.append(labels)

        accuracy = acc_meter.average

        outputs = {
            "accuracy": accuracy,
            "encoded_samples": encoded_samples,
            "loaded_labels": loaded_labels,
        }

        return outputs

    def personalized_train_one_epoch(
        self,
        epoch,
        config,
        pers_optimizer,
        lr_schedule,
        pers_loss_criterion,
        train_loader,
        epoch_loss_meter,
    ):
        # pylint:disable=too-many-arguments
        """Performing one epoch of learning for the personalization."""

        epoch_loss_meter.reset()
        self.personalized_model.train()
        self.personalized_model.to(self.device)
        pers_epochs = config["personalized_epochs"]

        local_progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{pers_epochs+1}", disable=True
        )

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            preds = self.personalized_model(examples)
            loss = pers_loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss, labels.size(0))

            local_progress.set_postfix(
                {
                    "lr": lr_schedule,
                    "loss": epoch_loss_meter.loss_value,
                    "loss_avg": epoch_loss_meter.average,
                }
            )

        return epoch_loss_meter

    def personalized_train_model(
        self,
        config,
        trainset,
        sampler,
        **kwargs,
    ):
        # pylint:disable=too-many-locals
        """The default training loop when a custom training loop is not supplied."""

        pers_epochs = config["personalized_epochs"]
        config["current_round"] = self.current_round
        batch_size = config["personalized_batch_size"]

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        assert "testset" in kwargs and "testset_sampler" in kwargs
        testset = kwargs["testset"]
        testset_sampler = kwargs["testset_sampler"]
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=testset_sampler.get()
        )

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the optimizer, lr_schedule, and loss criterion
        pers_optimizer = self.get_personalized_optimizer(self.personalized_model)
        pers_lr_scheduler = self.get_personalized_lr_scheduler(pers_optimizer)
        _pers_loss_criterion = self.get_personalized_loss_criterion()

        self.personalized_train_run_start(
            config,
            data_loader=test_loader,
        )

        self.personalized_model.to(self.device)

        # epoch loss tracker
        epoch_loss_meter = tracking.LossTracker()

        # Start personalization training
        # Note:
        #   To distanguish the eval training stage with the
        # previous ssl's training stage. We utilize the progress bar
        # to demonstrate the training progress details.
        show_str = f"[Client #{self.client_id}] Personalization"
        global_progress = tqdm(range(1, pers_epochs + 1), desc=show_str)

        for epoch in global_progress:
            self.personalized_train_epoch_start(config)

            epoch_loss_meter = self.personalized_train_one_epoch(
                epoch=epoch,
                config=config,
                pers_optimizer=pers_optimizer,
                lr_schedule=pers_lr_scheduler,
                pers_loss_criterion=_pers_loss_criterion,
                train_loader=self.train_loader,
                epoch_loss_meter=epoch_loss_meter,
            )

            pers_lr_scheduler.step()

            eval_outputs = self.personalized_train_epoch_end(
                epoch=epoch,
                config=config,
                data_loader=test_loader,
                loss=epoch_loss_meter.average,
            )

        # get the accuracy of the client
        accuracy = eval_outputs["accuracy"]
        self.personalized_train_run_end(config)

        # save the personalized model for current round
        # to the model dir of this client
        if "max_concurrency" in config:

            current_round = self.current_round

            personalized_model_name = config["personalized_model_name"]
            save_location = self.get_checkpoint_dir_path()
            filename = get_format_name(
                client_id=self.client_id,
                model_name=personalized_model_name,
                round_n=current_round,
                run_id=None,
                prefix="personalized",
                ext="pth",
            )
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename, location=save_location)

        if "max_concurrency" in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config["personalized_model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
            return None
        return accuracy

    def personalized_train_process(self, config, trainset, sampler, **kwargs):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: The sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.
        """

        try:
            self.personalized_train_model(config, trainset, sampler.get(), **kwargs)
        except Exception as training_exception:
            logging.info(
                "Personalization Training on client #%d failed.", self.client_id
            )
            raise training_exception

        if "max_concurrency" in config:
            self.personalized_model.cpu()
            model_type = config["personalized_model_name"]
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_personalized_model(filename)

    def personalized_train(self, trainset, sampler, **kwargs) -> float:
        """The main training loop in a federated learning workload for
            the personalization.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.

        Returns:
        float: Elapsed time during training.
        """

        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        accuracy = -1

        if "max_concurrency" in config:
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            train_proc = mp.Process(
                target=self.personalized_train_process,
                args=(config, trainset, sampler),
                kwargs=kwargs,
            )
            train_proc.start()
            train_proc.join()

            personalized_model_name = Config().trainer.personalized_model_name
            filename = (
                f"{personalized_model_name}_{self.client_id}_{config['run_id']}.pth"
            )

            acc_filename = (
                f"{personalized_model_name}_{self.client_id}_{config['run_id']}.acc"
            )
            try:
                self.load_personalized_model(filename)
                accuracy = self.load_accuracy(acc_filename)

            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Training on client {self.client_id} failed."
                ) from error

            toc = time.perf_counter()
            # self.pause_training()
        else:
            tic = time.perf_counter()
            self.personalized_train_process(config, trainset, sampler, **kwargs)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time, accuracy

    def personalized_train_run_start(self, config, **kwargs):
        """Method called at the start of training run.

        By default, we initial personalized model and its performance is recorded.
        """
        current_round = self.current_round
        data_loader = kwargs["data_loader"]

        eval_outputs = self.perform_evaluation_op(
            data_loader, defined_model=self.personalized_model
        )

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=eval_outputs["accuracy"],
            current_round=current_round,
            epoch=0,
            run_id=None,
        )

        return eval_outputs

    def personalized_train_run_end(self, config):
        """Method called at the end of a training run."""

    def personalized_train_epoch_start(self, config):
        """Method called at the beginning of a personalized training epoch."""

    def personalized_train_epoch_end(
        self,
        epoch,
        config,
        **kwargs,
    ):
        """The customize behavior after performing one epoch of personalized training.

        By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        data_loader = kwargs["data_loader"]
        loss = kwargs["loss"]
        pers_epochs = config["personalized_epochs"]

        current_round = self.current_round
        epoch_log_interval = pers_epochs + 1

        eval_outputs = {}
        if "personalized_epoch_log_interval" in config:
            epoch_log_interval = config["personalized_epoch_log_interval"]

        if epoch == 1 or epoch % epoch_log_interval == 0 or epoch == pers_epochs:
            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id,
                epoch,
                pers_epochs,
                loss,
            )

            eval_outputs = self.perform_evaluation_op(
                data_loader, self.personalized_model
            )
            accuracy = eval_outputs["accuracy"]

            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(
                accuracy=accuracy, current_round=current_round, epoch=epoch, run_id=None
            )

        return eval_outputs
