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

warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from plato.utils.checkpoint_operator import perform_client_checkpoint_saving
from plato.utils.checkpoint_operator import get_client_checkpoint_operator
from plato.utils.arrange_saving_name import get_format_name

from plato.utils import data_loaders_wrapper


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

    def set_client_personalized_model(self, personalized_model):
        """ Setting the client's personalized model """
        self.personalized_model = personalized_model

    @staticmethod
    def process_save_path(filename, location, work_model_name,
                          desired_extenstion):
        """ Process the input arguments to obtain the final saving path. """
        # default saving everything to the model path
        model_path = Config(
        ).params['model_path'] if location is None else location
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
            to_save_path = f'{model_path}/{filename}'
        else:
            to_save_path = f'{model_path}/{model_name}'

        # check the file extension
        save_prefix, save_extension = os.path.splitext(to_save_path)
        # the save file must contain a 'pth' as its extension
        if save_extension != desired_extenstion:
            to_save_path = save_prefix + desired_extenstion

        return to_save_path

    def save_personalized_model(self, filename=None, location=None):
        """ Saving the model to a file. """
        # process the arguments to obtain the to save path
        to_save_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth")

        if self.personalized_model_state_dict is None:
            torch.save(self.personalized_model.state_dict(), to_save_path)
        else:
            torch.save(self.personalized_model_state_dict, to_save_path)

        logging.info("[Client #%d] Personalized Model saved to %s.",
                     self.client_id, to_save_path)

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        # process the arguments to obtain the to save path
        load_from_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth")

        logging.info("[Client #%d] Loading a Personalized model from %s.",
                     self.client_id, load_from_path)

        self.personalized_model.load_state_dict(torch.load(load_from_path),
                                                strict=True)

    @staticmethod
    def save_personalized_accuracy(accuracy,
                                   round=None,
                                   epoch=None,
                                   accuracy_type="monitor_accuracy",
                                   filename=None,
                                   location=None):
        """Saving the test accuracy to a file."""
        to_save_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".csv")

        current_round = round if round is not None else 0
        current_epoch = epoch if epoch is not None else 0
        acc_dataframe = pd.DataFrame(
            {
                "round": current_round,
                "epoch": current_epoch,
                accuracy_type: accuracy
            },
            index=[0])

        is_use_header = True if not os.path.exists(
            to_save_accuracy_path) else False
        acc_dataframe.to_csv(to_save_accuracy_path,
                             index=False,
                             mode='a',
                             header=is_use_header)

    @staticmethod
    def load_personalized_accuracy(round=None,
                                   accuracy_type="monitor_accuracy",
                                   filename=None,
                                   location=None):
        """Loading the test accuracy from a file."""
        to_load_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".acc")

        loaded_rounds_accuracy = pd.read_csv(to_load_accuracy_path)
        if round is None:
            # default use the last row
            desired_row = loaded_rounds_accuracy.iloc[-1]
        else:
            desired_row = loaded_rounds_accuracy.loc[
                loaded_rounds_accuracy['round'] == round]
            desired_row = loaded_rounds_accuracy.iloc[-1]

        accuracy = desired_row[accuracy_type]

        return accuracy

    def checkpoint_personalized_accuracy(self, accuracy, current_round, epoch,
                                         run_id):
        # save the personaliation accuracy to the results dir
        result_path = Config().params['result_path']

        save_location = os.path.join(result_path,
                                     "client_" + str(self.client_id))

        filename = get_format_name(client_id=self.client_id,
                                   run_id=run_id,
                                   suffix="personalization",
                                   ext="csv")
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_accuracy(
            accuracy,
            round=current_round,
            epoch=epoch,
            accuracy_type="personalization_accuracy",
            filename=filename,
            location=save_location)

    def save_encoded_data(self,
                          encoded_data,
                          data_labels,
                          filename=None,
                          location=None):
        """ Save the encoded data (np.narray). """
        # convert the list to tensor
        encoded_data = torch.cat(encoded_data, axis=0)
        data_labels = torch.cat(data_labels)
        # combine the label to the final of the 2d
        to_save_data = torch.cat(
            [encoded_data, data_labels.reshape(-1, 1)], dim=1)
        to_save_narray_data = to_save_data.detach().to("cpu").numpy()

        # process the arguments to obtain the to save path
        to_save_path = self.process_save_path(filename,
                                              location,
                                              work_model_name="model_name",
                                              desired_extenstion=".npy")

        np.save(to_save_path, to_save_narray_data, allow_pickle=True)
        logging.info("[Client #%d] Saving encoded data to %s.", self.client_id,
                     to_save_path)

    def checkpoint_encoded_samples(self,
                                   encoded_samples,
                                   encoded_labels,
                                   current_round,
                                   epoch,
                                   run_id,
                                   encoded_type="trainEncoded"):

        # save the encoded data to the results dir
        result_path = Config().params['result_path']
        save_location = os.path.join(result_path,
                                     "client_" + str(self.client_id))
        save_filename = get_format_name(client_id=self.client_id,
                                        round_n=current_round,
                                        epoch_n=epoch,
                                        run_id=run_id,
                                        suffix=encoded_type,
                                        ext="npy")

        self.save_encoded_data(encoded_data=encoded_samples,
                               data_labels=encoded_labels,
                               filename=save_filename,
                               location=save_location)

    def prepare_train_lr(self, optimizer, train_data_loader, config,
                         current_round):
        """ Prepare the lr schedule for training process.

            The obtained lr should be scheduled in the whole learning
            process, i.e., communication_rounds * local_epochs
        """

        epochs = config['epochs']
        iterations_per_epoch = len(train_data_loader)
        # Note, the lr_schedule_base_epoch is an important term to make the
        # lr_schedulr work correctly.
        # The main reason is that the trainer will create a new lr schedule
        # in each round. Then, the epoch within one round will always start
        # from 0. Therefore, if the lr schedule works based this local epoch,
        # the lr will never be modified correctly as that in the central learning.
        # Thus, we need a term to denote the global epoch.
        # In round 1, the base global epoch should be 0. Thus, the local epoch
        # can start from 1 * 0 to epochs, i.e., [0, epochs].
        # In round 2, the base global epoch should be 'epochs'. Thus, the local epoch
        # can start from 1 * 'epochs' to epochs + 'epochs', i.e, [epochs, epochs + epochs]
        # Then, in round r, the base global epoch should be (current_round - 1) * epochs
        # Therefore, the local epoch for the lr schedule can:
        #   start from lr_schedule_base_epoch + 0,
        #   to
        #   lr_schedule_base_epoch + epochs
        lr_schedule_base_epoch = (current_round - 1) * epochs

        # Initializing the learning rate schedule, if necessary
        lr_schedule = optimizers.get_dynamic_lr_schedule(
            optimizer, iterations_per_epoch, train_data_loader)

        # Updated the lr_schedule to the latest status
        if lr_schedule_base_epoch != 0:
            for _ in range(1, lr_schedule_base_epoch + 1):
                lr_schedule.step()

        return lr_schedule, lr_schedule_base_epoch

    def train_one_epoch(self, config, epoch, defined_model, optimizer,
                        loss_criterion, train_data_loader, epoch_loss_meter,
                        batch_loss_meter):
        defined_model.train()
        epochs = config['epochs']
        iterations_per_epoch = len(train_data_loader)
        # default not to perform any logging
        epoch_log_interval = epochs + 1
        batch_log_interval = iterations_per_epoch

        if "epoch_log_interval" in config:
            epoch_log_interval = config['epoch_log_interval']
        if "batch_log_interval" in config:
            batch_log_interval = config['batch_log_interval']

        epoch_loss_meter.reset()
        # Use a default training loop
        for batch_id, (examples, labels) in enumerate(train_data_loader):
            # Support a more general way to hold the loaded samples
            # The defined model is responsible for processing the
            # examples based on its requirements.
            if torch.is_tensor(examples):
                examples = examples.to(self.device)
            else:
                examples = [
                    each_sample.to(self.device) for each_sample in examples
                ]

            labels = labels.to(self.device)

            # Reset and clear previous data
            batch_loss_meter.reset()
            optimizer.zero_grad()

            # Forward the model and compute the loss
            outputs = defined_model(examples)
            loss = loss_criterion(outputs, labels)

            # Perform the backpropagation
            loss.backward()
            optimizer.step()

            # Update the loss data in the logging container
            epoch_loss_meter.update(loss.data.item(), labels.size(0))
            batch_loss_meter.update(loss.data.item(), labels.size(0))

            # Performe logging of one batch
            if batch_id % batch_log_interval == 0 or batch_id == iterations_per_epoch - 1:
                if self.client_id == 0:
                    logging.info(
                        "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                        os.getpid(), epoch, epochs, batch_id,
                        iterations_per_epoch - 1, batch_loss_meter.avg)
                else:
                    logging.info(
                        "   [Client #%d] Training Epoch: \
                        [%d/%d][%d/%d]\tLoss: %.6f", self.client_id, epoch,
                        epochs, batch_id, iterations_per_epoch - 1,
                        batch_loss_meter.avg)

        # Performe logging of epochs
        if (epoch - 1) % epoch_log_interval == 0 or epoch == epochs:
            logging.info("[Client #%d] Training Epoch: [%d/%d]\tLoss: %.6f",
                         self.client_id, epoch, epochs, epoch_loss_meter.avg)

        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

    def train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default personalized training loop when a custom training loop is not supplied.

        """
        # Customize the config
        config = self.customize_train_config(config)

        batch_size = config['batch_size']
        model_type = config['model_name']
        current_round = kwargs['current_round']
        run_id = config['run_id']
        # Obtain the logging interval
        epochs = config['epochs']
        config['current_round'] = current_round

        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        # to get the specific sampler, Plato's sampler should perform
        # Sampler.get()
        # However, for train's ssampler, the self.sampler.get() has been
        # performed within the train_process of the trainer/basic.py
        # Thus, there is no need to further perform .get() here.
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   sampler=sampler)

        # obtain the loader for unlabeledset if possible
        # unlabeled_trainset, unlabeled_sampler
        unlabeled_loader = None
        unlabeled_trainset = []
        if "unlabeled_trainset" in kwargs and kwargs[
                "unlabeled_trainset"] is not None:
            unlabeled_trainset = kwargs["unlabeled_trainset"]
            unlabeled_sampler = kwargs["unlabeled_sampler"]
            unlabeled_loader = torch.utils.data.DataLoader(
                dataset=unlabeled_trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=unlabeled_sampler.get())

        # wrap the multiple loaders into one sequence loader
        streamed_train_loader = data_loaders_wrapper.StreamBatchesLoader(
            [train_loader, unlabeled_loader])

        epoch_model_log_interval = epochs + 1
        if "epoch_model_log_interval" in config:
            epoch_model_log_interval = config['epoch_model_log_interval']

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model, config)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        _optimizer_func = getattr(self, "get_optimizer", None)
        if callable(_optimizer_func):
            optimizer = self.get_optimizer(self.model, config)
        else:
            optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        lr_schedule, lr_schedule_base_epoch = self.prepare_train_lr(
            optimizer, streamed_train_loader, config, current_round)

        logging.info(
            f"With {lr_schedule}, we get lr={lr_schedule.get_lr()} under the global epoch {lr_schedule_base_epoch}"
        )

        # Before the training, we expect to save the initial
        # model of this round
        # this is determinted by whether to perform the detailed
        # checkpoints of the training models
        # if 'do_detailed_checkpoint' in config and config[
        #         'do_detailed_checkpoint']:
        #     perform_client_checkpoint_saving(
        #         client_id=self.client_id,
        #         model_name=model_type,
        #         model_state_dict=self.model.state_dict(),
        #         config=config,
        #         optimizer_state_dict=optimizer.state_dict(),
        #         lr_schedule_state_dict=lr_schedule.state_dict(),
        #         present_epoch=0,
        #         base_epoch=lr_schedule_base_epoch)

        # Sending the model to the device used for training
        self.model.to(self.device)

        # Define the container to hold the logging information
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        # Start training
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.train_one_epoch(config,
                                 epoch,
                                 defined_model=self.model,
                                 optimizer=optimizer,
                                 loss_criterion=loss_criterion,
                                 train_data_loader=streamed_train_loader,
                                 epoch_loss_meter=epoch_loss_meter,
                                 batch_loss_meter=batch_loss_meter)

            # Update the learning rate
            # based on the base epoch
            lr_schedule.step()
            # this is determinted by whether to perform the detailed
            # checkpoints of the training models
            # if 'do_detailed_checkpoint' in config and config[
            #         'do_detailed_checkpoint']:
            #     if (epoch -
            #             1) % epoch_model_log_interval == 0 or epoch == epochs:
            #         # the model generated during each round will be stored in the
            #         # checkpoints
            #         perform_client_checkpoint_saving(
            #             client_id=self.client_id,
            #             model_name=model_type,
            #             model_state_dict=self.model.state_dict(),
            #             config=config,
            #             optimizer_state_dict=optimizer.state_dict(),
            #             lr_schedule_state_dict=lr_schedule.state_dict(),
            #             present_epoch=epoch,
            #             base_epoch=lr_schedule_base_epoch + epoch)

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

        if 'max_concurrency' in config:
            # the final of each round, the trained model within this round
            # will be saved as model to the '/models' dir

            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=model_type,
                model_state_dict=self.model.state_dict(),
                config=config,
                optimizer_state_dict=optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=None,
                base_epoch=lr_schedule_base_epoch + epochs)

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
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

        accuracy = acc_meter.avg

        outputs = {"accuracy": accuracy}

        return outputs

    def on_start_pers_train(
        self,
        defined_model,
        model_name,
        data_loader,
        epoch,
        global_epoch,
        config,
        optimizer,
        lr_schedule,
        **kwargs,
    ):
        """ The customize behavior before performing one epoch of personalized training.

            By default, we need to save the the accuracy, and the model when possible.
        """
        current_round = config['current_round']
        save_checkpoint_filename = perform_client_checkpoint_saving(
            client_id=self.client_id,
            model_name=model_name,
            model_state_dict=defined_model.state_dict(),
            config=config,
            optimizer_state_dict=optimizer.state_dict()
            if optimizer is not None else None,
            lr_schedule_state_dict=lr_schedule.state_dict()
            if lr_schedule is not None else None,
            present_epoch=epoch,
            base_epoch=global_epoch,
            prefix="personalized")

        eval_outputs = self.perform_evaluation_op(data_loader, defined_model)

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=eval_outputs["accuracy"],
            current_round=current_round,
            epoch=epoch,
            run_id=None)

        return eval_outputs, save_checkpoint_filename

    def on_end_pers_train_epoch(
        self,
        defined_model,
        model_name,
        data_loader,
        epoch,
        global_epoch,
        config,
        optimizer,
        lr_schedule,
        epoch_loss_meter,
        **kwargs,
    ):
        """ The customize behavior after performing one epoch of personalized training.

            By default, we need to save the encoded data, the accuracy, and the model when possible.
        """

        pers_epochs = config["pers_epochs"]
        current_round = config['current_round']
        epoch_log_interval = pers_epochs + 1
        epoch_model_log_interval = pers_epochs + 1
        eval_outputs = {}
        if "pers_epoch_log_interval" in config:
            epoch_log_interval = config['pers_epoch_log_interval']

        if "pers_epoch_model_log_interval" in config:
            epoch_model_log_interval = config['pers_epoch_model_log_interval']

        if (epoch - 1) % epoch_log_interval == 0 or epoch == pers_epochs:
            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id, epoch, pers_epochs, epoch_loss_meter.avg)

            eval_outputs = self.perform_evaluation_op(data_loader,
                                                      defined_model)
            accuracy = eval_outputs["accuracy"]

            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(accuracy=accuracy,
                                                  current_round=current_round,
                                                  epoch=epoch,
                                                  run_id=None)

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
                base_epoch=global_epoch,
                prefix="personalized")

        return eval_outputs

    def pers_train_one_epoch(
        self,
        config,
        epoch,
        defined_model,
        pers_optimizer,
        lr_schedule,
        pers_loss_criterion,
        pers_train_loader,
        epoch_loss_meter,
    ):
        """ Performing one epoch of learning for the personalization. """

        epoch_loss_meter.reset()
        defined_model.train()
        defined_model.to(self.device)

        pers_epochs = config["pers_epochs"]

        local_progress = tqdm(pers_train_loader,
                              desc=f'Epoch {epoch}/{pers_epochs+1}',
                              disable=True)

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            preds = defined_model(examples)
            loss = pers_loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss.data.item(), labels.size(0))

            local_progress.set_postfix({
                'lr': lr_schedule,
                "loss": epoch_loss_meter.val,
                'loss_avg': epoch_loss_meter.avg
            })

        return epoch_loss_meter

    def pers_train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

        """
        # Customize the config
        config = self.customize_train_config(config)

        current_round = kwargs['current_round']
        personalized_model_name = config['personalized_model_name']
        config['current_round'] = current_round

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            assert "testset" in kwargs and "testset_sampler" in kwargs
            testset = kwargs['testset']
            testset_sampler = kwargs['testset_sampler']
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['pers_batch_size'],
                shuffle=False,
                sampler=testset_sampler.get())

            pers_train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config['pers_batch_size'],
                shuffle=False,
                sampler=sampler.get())

            # Perform the personalization
            #   i.e., the client's personal local dataset
            pers_optimizer = optimizers.get_dynamic_optimizer(
                self.personalized_model, prefix="pers_")
            iterations_per_epoch = len(pers_train_loader)

            # Initializing the learning rate schedule, if necessary
            assert 'pers_lr_schedule' in config
            lr_schedule = optimizers.get_dynamic_lr_schedule(
                optimizer=pers_optimizer,
                iterations_per_epoch=iterations_per_epoch,
                train_loader=pers_train_loader,
                prefix="pers_")

            # Initializing the loss criterion
            _pers_loss_criterion = getattr(self, "pers_loss_criterion", None)
            if callable(_pers_loss_criterion):
                pers_loss_criterion = self.pers_loss_criterion(
                    self.personalized_model)
            else:
                pers_loss_criterion = torch.nn.CrossEntropyLoss()

            self.personalized_model.to(self.device)

            # Define the training and logging information
            # default not to perform any logging
            pers_epochs = config['pers_epochs']

            # epoch loss tracker
            epoch_loss_meter = optimizers.AverageMeter(name='Loss')

            # Start personalization training
            # Note:
            #   To distanguish the eval training stage with the
            # previous ssl's training stage. We utilize the progress bar
            # to demonstrate the training progress details.
            global_progress = tqdm(range(1, pers_epochs + 1),
                                   desc='Personalization')
            self.on_start_pers_train(
                defined_model=self.personalized_model,
                model_name=personalized_model_name,
                data_loader=test_loader,
                epoch=0,
                global_epoch=0,
                config=config,
                optimizer=pers_optimizer,
                lr_schedule=lr_schedule,
            )

            for epoch in global_progress:

                epoch_loss_meter = self.pers_train_one_epoch(
                    config=config,
                    epoch=epoch,
                    defined_model=self.personalized_model,
                    pers_optimizer=pers_optimizer,
                    lr_schedule=lr_schedule,
                    pers_loss_criterion=pers_loss_criterion,
                    pers_train_loader=pers_train_loader,
                    epoch_loss_meter=epoch_loss_meter)

                lr_schedule.step()

                eval_outputs = self.on_end_pers_train_epoch(
                    defined_model=self.personalized_model,
                    model_name=personalized_model_name,
                    data_loader=test_loader,
                    epoch=epoch,
                    global_epoch=epoch,
                    config=config,
                    optimizer=pers_optimizer,
                    lr_schedule=lr_schedule,
                    epoch_loss_meter=epoch_loss_meter)

        except Exception as testing_exception:
            logging.info("Personalization Learning on client #%d failed.",
                         self.client_id)
            raise testing_exception

        # get the accuracy of the client
        accuracy = eval_outputs["accuracy"]

        # save the personalized model for current round
        # to the model dir of this client
        if 'max_concurrency' in config:

            current_round = kwargs['current_round']
            if current_round == config['rounds']:
                target_dir = Config().params['model_path']
            else:
                target_dir = Config().params['checkpoint_path']
            personalized_model_name = config['personalized_model_name']
            save_location = os.path.join(target_dir,
                                         "client_" + str(self.client_id))
            filename = get_format_name(client_id=self.client_id,
                                       model_name=personalized_model_name,
                                       round_n=current_round,
                                       run_id=None,
                                       prefix="personalized",
                                       ext="pth")
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename,
                                         location=save_location)

        if 'max_concurrency' in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config['personalized_model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)

        else:
            return accuracy

    def pers_train_process(self,
                           config,
                           trainset,
                           sampler,
                           cut_layer=None,
                           **kwargs):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: The sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        kwargs (optional): Additional keyword arguments.
        """

        try:
            self.pers_train_model(config, trainset, sampler, cut_layer,
                                  **kwargs)
        except Exception as training_exception:
            logging.info("Personalization Training on client #%d failed.",
                         self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.personalized_model.cpu()
            model_type = config['personalized_model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_personalized_model(filename)

    def pers_train(self, trainset, sampler, cut_layer=None, **kwargs) -> float:
        """The main training loop in a federated learning workload for
            the personalization.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        kwargs (optional): Additional keyword arguments.

        Returns:
        float: Elapsed time during training.
        """

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        accuracy = -1

        if 'max_concurrency' in config:
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            train_proc = mp.Process(target=self.pers_train_process,
                                    args=(config, trainset, sampler,
                                          cut_layer),
                                    kwargs=kwargs)
            train_proc.start()
            train_proc.join()

            personalized_model_name = Config().trainer.personalized_model_name
            filename = f"{personalized_model_name}_{self.client_id}_{config['run_id']}.pth"

            acc_filename = f"{personalized_model_name}_{self.client_id}_{config['run_id']}.acc"
            try:
                self.load_personalized_model(filename)
                accuracy = self.load_accuracy(acc_filename)

            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Training on client {self.client_id} failed.") from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.pers_train_process(config, trainset, sampler, cut_layer,
                                    **kwargs)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time, accuracy

    def customize_train_config(self, config):
        """ Customize the training config based on the user's own requirement. """

        # By default, we save all checkpoints in the personalization stage.
        # this can also be regarded as a demo of how to change the trainer's config.
        config['do_detailed_pers_checkpoint'] = False
        return config
