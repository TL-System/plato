"""
The training and testing loops for PyTorch.

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

    However, for the self.personalized model, we directly save it as model.state_dict()
    instead of the checkpoint file.
        -> Therefore, loading the model parameter of the checkpoint file should be:
        torch.load(filepath)


    These settings induce our recall self.model and self.personalized_model in each round
    Please access the 'load_personalized_model' and 'load_payload' in the
    clients/pers_simple.py.

"""

import logging
import multiprocessing as mp
import os
import time
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import loss_criterion
import optimizers
import lr_schedulers
from plato.config import Config
from plato.trainers import basic
import tracking
from checkpoint_operator import perform_client_checkpoint_saving
from arrange_saving_name import get_format_name


class Trainer(basic.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""

    def __init__(self, model=None):
        super().__init__(model)

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

    def get_loss_criterion(self, stage_prefix=None):
        """Returns the loss criterion."""
        return loss_criterion.get(stage_prefix)

    def get_optimizer(self, model, stage_prefix=None):
        """Returns the optimizer."""
        return optimizers.get(model, stage_prefix)

    def get_lr_scheduler(self, config, optimizer, stage_prefix=None):
        """Returns the learning rate scheduler, if needed."""
        if "lr_scheduler" not in config:
            return None

        return lr_schedulers.get(
            optimizer, len(self.train_loader), stage_prefix=stage_prefix
        )

    @staticmethod
    def process_save_path(filename, location, work_model_name, desired_extenstion):
        """Process the input arguments to obtain the final saving path."""
        # default saving everything to the model path
        model_path = Config().params["model_path"] if location is None else location
        # set the model_type to
        #  - "model_name" to obtain the global model's name
        #  - "personalized_model_name" to obtain the personalized model's name
        #  - or any other model you defined and want to process
        model_name = getattr(Config().trainer, work_model_name)
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            to_save_path = f"{model_path}/{filename}"
        else:
            to_save_path = f"{model_path}/{model_name}"

        # check the file extension
        save_prefix, save_extension = os.path.splitext(to_save_path)
        # the save file must contain a 'pth' as its extension
        if save_extension != desired_extenstion:
            to_save_path = save_prefix + desired_extenstion

        return to_save_path

    def save_personalized_model(self, filename=None, location=None):
        """Saving the model to a file."""
        # process the arguments to obtain the to save path
        to_save_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth",
        )

        if self.personalized_model_state_dict is None:
            torch.save(self.personalized_model.state_dict(), to_save_path)
        else:
            torch.save(self.personalized_model_state_dict, to_save_path)

        logging.info(
            "[Client #%d] Personalized Model saved to %s.", self.client_id, to_save_path
        )

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        # process the arguments to obtain the to save path
        load_from_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth",
        )

        logging.info(
            "[Client #%d] Loading a Personalized model from %s.",
            self.client_id,
            load_from_path,
        )

        self.personalized_model.load_state_dict(torch.load(load_from_path), strict=True)

    @staticmethod
    def save_personalized_accuracy(
        accuracy,
        round=None,
        epoch=None,
        accuracy_type="monitor_accuracy",
        filename=None,
        location=None,
    ):
        """Saving the test accuracy to a file."""
        to_save_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".csv",
        )

        current_round = round if round is not None else 0
        current_epoch = epoch if epoch is not None else 0
        acc_dataframe = pd.DataFrame(
            {"round": current_round, "epoch": current_epoch, accuracy_type: accuracy},
            index=[0],
        )

        is_use_header = True if not os.path.exists(to_save_accuracy_path) else False
        acc_dataframe.to_csv(
            to_save_accuracy_path, index=False, mode="a", header=is_use_header
        )

    @staticmethod
    def load_personalized_accuracy(
        round=None, accuracy_type="monitor_accuracy", filename=None, location=None
    ):
        """Loading the test accuracy from a file."""
        to_load_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".acc",
        )

        loaded_rounds_accuracy = pd.read_csv(to_load_accuracy_path)
        if round is None:
            # default use the last row
            desired_row = loaded_rounds_accuracy.iloc[-1]
        else:
            desired_row = loaded_rounds_accuracy.loc[
                loaded_rounds_accuracy["round"] == round
            ]
            desired_row = loaded_rounds_accuracy.iloc[-1]

        accuracy = desired_row[accuracy_type]

        return accuracy

    def checkpoint_personalized_accuracy(self, accuracy, current_round, epoch, run_id):
        # save the personaliation accuracy to the results dir
        result_path = Config().params["result_path"]

        save_location = os.path.join(result_path, "client_" + str(self.client_id))

        filename = get_format_name(
            client_id=self.client_id, run_id=run_id, suffix="personalization", ext="csv"
        )
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_accuracy(
            accuracy,
            round=current_round,
            epoch=epoch,
            accuracy_type="personalization_accuracy",
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
        to_save_path = self.process_save_path(
            filename, location, work_model_name="model_name", desired_extenstion=".npy"
        )

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

        # save the encoded data to the results dir
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

    def train_step_start(self, config, batch_samples=None, batch_labels=None):
        """
        Method called at the start of a training step.

        :param batch_samples: the current batch of samples.
        :param batch_labels: the current batch of labesl.
        """
        if torch.is_tensor(batch_samples):
            batch_samples = batch_samples.to(self.device)
        else:
            batch_samples = [
                each_sample.to(self.device) for each_sample in batch_samples
            ]

        batch_labels = batch_labels.to(self.device)

        return batch_samples, batch_labels

    def train_one_epoch(
        self, config, optimizer, loss_criterion, train_data_loader, epoch_loss_meter
    ):
        """Perform one epoch of training."""
        self.model.train()
        epoch_loss_meter.reset()

        # Use a default training loop
        for batch_id, (examples, labels) in enumerate(train_data_loader):
            examples, labels = self.train_step_start(
                config, batch_samples=examples, batch_labels=labels
            )

            # Reset and clear previous data
            optimizer.zero_grad()

            # Forward the model and compute the loss
            outputs = self.model(examples)
            loss = loss_criterion(outputs, labels)

            # Perform the backpropagation
            if "create_graph" in config:
                loss.backward(create_graph=config["create_graph"])
            else:
                loss.backward()

            optimizer.step()

            # Update the loss data in the logging container
            epoch_loss_meter.update(loss, labels.size(0))

            self.train_step_end(config, batch=batch_id, loss=loss)
            self.callback_handler.call_event(
                "on_train_step_end", self, config, batch=batch_id, loss=loss
            )

        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

    def train_model(
        self,
        config,
        trainset,
        sampler,
        **kwargs,
    ):
        """The default personalized training loop when a custom training loop is not supplied."""
        # Customize the config
        config = self.customize_train_config(config)
        batch_size = config["batch_size"]
        model_type = config["model_name"]

        epochs = config["epochs"]
        config["current_round"] = self.current_round

        self.sampler = sampler

        self.run_history.reset()
        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        _loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, optimizer)
        optimizer = self._adjust_lr(config, self.lr_scheduler, optimizer)

        # Sending the model to the device used for training
        self.model.to(self.device)

        # Start training
        for self.current_epoch in range(1, epochs + 1):
            self.model.train()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            self.train_one_epoch(
                config,
                optimizer=optimizer,
                loss_criterion=_loss_criterion,
                train_data_loader=self.train_loader,
                epoch_loss_meter=self._loss_tracker,
            )

            self.lr_scheduler.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

        if "max_concurrency" in config:
            # the final of each round, the trained model within this round
            # will be saved as model to the '/models' dir
            model_type = model_type.replace("/", "_")
            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=model_type,
                model_state_dict=self.model.state_dict(),
                config=config,
                optimizer_state_dict=optimizer.state_dict(),
                lr_schedule_state_dict=_loss_criterion.state_dict(),
                present_epoch=None,
                base_epoch=(self.current_round - 1) * epochs + epochs,
            )

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):
        """The operation of performing the evaluation on the testset with the defined model."""
        # Define the test phase of the eval stage
        acc_meter = tracking.AverageMeter(name="Accuracy")
        defined_model.eval()
        defined_model.to(self.device)

        correct = 0

        acc_meter.reset()
        for _, (examples, labels) in enumerate(to_eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                preds = defined_model(examples).argmax(dim=1)
                correct = (preds == labels).sum().item()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

        accuracy = acc_meter.average

        outputs = {"accuracy": accuracy}

        return outputs

    def pers_train_one_epoch(
        self,
        config,
        pers_optimizer,
        lr_schedule,
        loss_criterion,
        train_loader,
        epoch_loss_meter,
    ):
        """Performing one epoch of learning for the personalization."""

        epoch_loss_meter.reset()
        self.personalized_model.train()
        self.personalized_model.to(self.device)
        epoch = self.current_epoch
        pers_epochs = config["pers_epochs"]

        local_progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{pers_epochs+1}", disable=True
        )

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = self.train_step_start(
                config, batch_samples=examples, batch_labels=labels
            )

            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            preds = self.personalized_model(examples)
            loss = loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss, labels.size(0))

            local_progress.set_postfix(
                {
                    "lr": lr_schedule,
                    "loss": epoch_loss_meter.val,
                    "loss_avg": epoch_loss_meter.average,
                }
            )

        return epoch_loss_meter

    def pers_train_model(
        self,
        config,
        trainset,
        sampler,
        **kwargs,
    ):
        """The default training loop when a custom training loop is not supplied."""
        # Customize the config
        config = self.customize_train_config(config)
        pers_epochs = config["pers_epochs"]
        personalized_model_name = config["personalized_model_name"]
        config["current_round"] = self.current_round
        batch_size = config["pers_batch_size"]

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
        pers_optimizer = self.get_optimizer(
            self.personalized_model, stage_prefix="pers"
        )
        pers_lr_scheduler = self.get_lr_scheduler(
            config, pers_optimizer, stage_prefix="pers"
        )
        _pers_loss_criterion = self.get_loss_criterion(stage_prefix="pers")

        self.pers_train_run_start(
            config,
            model_name=personalized_model_name,
            optimizer=pers_optimizer,
            lr_schedule=pers_lr_scheduler,
            data_loader=test_loader,
        )

        self.personalized_model.to(self.device)

        # epoch loss tracker
        epoch_loss_meter = tracking.AverageMeter(name="Loss")

        # Start personalization training
        # Note:
        #   To distanguish the eval training stage with the
        # previous ssl's training stage. We utilize the progress bar
        # to demonstrate the training progress details.
        show_str = f"[Client #{self.client_id}] Personalization"
        global_progress = tqdm(range(1, pers_epochs + 1), desc=show_str)

        for self.current_epoch in global_progress:
            self.pers_train_epoch_start(config)

            epoch_loss_meter = self.pers_train_one_epoch(
                config=config,
                pers_optimizer=pers_optimizer,
                lr_schedule=pers_lr_scheduler,
                loss_criterion=_pers_loss_criterion,
                train_loader=self.train_loader,
                epoch_loss_meter=epoch_loss_meter,
            )

            pers_lr_scheduler.step()

            eval_outputs = self.pers_train_epoch_end(
                config=config,
                model_name=personalized_model_name,
                data_loader=test_loader,
                optimizer=pers_optimizer,
                lr_schedule=pers_lr_scheduler,
                loss=epoch_loss_meter.average,
            )

        # get the accuracy of the client
        accuracy = eval_outputs["accuracy"]
        self.pers_train_run_end(config)

        # save the personalized model for current round
        # to the model dir of this client
        if "max_concurrency" in config:

            current_round = self.current_round
            if current_round == config["rounds"]:
                target_dir = Config().params["model_path"]
            else:
                target_dir = Config().params["checkpoint_path"]
            personalized_model_name = config["personalized_model_name"]
            save_location = os.path.join(target_dir, "client_" + str(self.client_id))
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

        else:
            return accuracy

    def pers_train_process(self, config, trainset, sampler, **kwargs):
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
            self.pers_train_model(config, trainset, sampler.get(), **kwargs)
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

    def pers_train(self, trainset, sampler, **kwargs) -> float:
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
                target=self.pers_train_process,
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
            self.pers_train_process(config, trainset, sampler, **kwargs)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time, accuracy

    def customize_train_config(self, config):
        """Customize the training config based on the user's own requirement."""

        # By default, we save all checkpoints in the personalization stage.
        # this can also be regarded as a demo of how to change the trainer's config.
        config["do_detailed_pers_checkpoint"] = False
        return config

    def pers_train_run_start(self, config, **kwargs):
        """Method called at the start of training run.

        By default, we initial personalized model and its performance is recorded.
        """
        current_round = self.current_round
        model_name = config["model_name"]
        optimizer = kwargs["optimizer"]
        lr_schedule = kwargs["lr_schedule"]
        data_loader = kwargs["data_loader"]
        model_name = model_name.replace("/", "_")
        save_checkpoint_filename = perform_client_checkpoint_saving(
            client_id=self.client_id,
            model_name=model_name,
            model_state_dict=self.personalized_model.state_dict(),
            config=config,
            optimizer_state_dict=optimizer.state_dict()
            if optimizer is not None
            else None,
            lr_schedule_state_dict=lr_schedule.state_dict()
            if lr_schedule is not None
            else None,
            present_epoch=self.current_epoch,
            base_epoch=self.current_epoch,
            prefix="personalized",
        )

        eval_outputs = self.perform_evaluation_op(
            data_loader, defined_model=self.personalized_model
        )

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=eval_outputs["accuracy"],
            current_round=current_round,
            epoch=self.current_epoch,
            run_id=None,
        )

        return eval_outputs, save_checkpoint_filename

    def pers_train_run_end(self, config):
        """Method called at the end of a training run."""

    def pers_train_epoch_start(self, config):
        """Method called at the beginning of a personalized training epoch."""

    def pers_train_epoch_end(
        self,
        config,
        **kwargs,
    ):
        """The customize behavior after performing one epoch of personalized training.

        By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        model_name = config["model_name"]
        optimizer = kwargs["optimizer"]
        lr_schedule = kwargs["lr_schedule"]
        data_loader = kwargs["data_loader"]
        loss = kwargs["loss"]
        pers_epochs = config["pers_epochs"]
        epoch = self.current_epoch

        current_round = self.current_round
        epoch_log_interval = pers_epochs + 1
        epoch_model_log_interval = pers_epochs + 1
        eval_outputs = {}
        if "pers_epoch_log_interval" in config:
            epoch_log_interval = config["pers_epoch_log_interval"]

        if "pers_epoch_model_log_interval" in config:
            epoch_model_log_interval = config["pers_epoch_model_log_interval"]

        if (epoch - 1) % epoch_log_interval == 0 or epoch == pers_epochs:
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

        model_name = model_name.replace("/", "_")
        # Whether to store the checkpoints
        if (epoch - 1) % epoch_model_log_interval == 0 or epoch == pers_epochs:
            # the model generated during each round will be stored in the
            # checkpoints
            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=model_name,
                model_state_dict=self.personalized_model.state_dict(),
                config=config,
                optimizer_state_dict=optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=epoch,
                base_epoch=self.current_epoch * self.current_round,
                prefix="personalized",
            )

        return eval_outputs
