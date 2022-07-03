"""
Implement the trainer for contrastive self-supervised learning method.

"""

import os
import logging
import time
import multiprocessing as mp

import numpy as np
import torch

from tqdm import tqdm
import pandas as pd

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from plato.utils import data_loaders_wrapper

from plato.models import ssl_monitor_register

from plato.utils import ssl_losses, checkpoint_saver
from plato.utils.arrange_saving_name import client_get_name


class Trainer(basic.Trainer):
    """ A federated learning trainer for contrastive self-supervised methods. """

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
    def loss_criterion(model):
        """ The loss computation.
            Currently, we only support the NT_Xent.

            The pytorch_metric_learning provides a strong
            support for loss criterion. However, how to use
            its NTXent is still nor clear.
            The loss criterion will be replaced by the one
            in pytorch_metric_learning afterward.
        """
        # define the loss computation instance
        defined_temperature = Config().trainer.temperature

        criterion = ssl_losses.NTXent(defined_temperature, world_size=1)

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            encoded_z1, encoded_z2 = outputs
            loss = criterion(encoded_z1, encoded_z2)
            return loss

        return loss_compute

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

        # check the filr extension
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
        acc_dataframe = pd.DataFrame(
            {
                "round": current_round,
                accuracy_type: accuracy
            }, index=[0])

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

    def get_checkppint_saver(self):
        target_dir = Config().params['checkpoint_path']
        to_save_dir = os.path.join(target_dir, "client_" + str(self.client_id))
        cpk_saver = checkpoint_saver.CheckpointsSaver(
            checkpoints_dir=to_save_dir)
        return cpk_saver

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

    def train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

        Note:
            This is the training stage of self-supervised learning (ssl). It is responsible
        for performing the contrastive learning process based on the trainset to train
        a encoder in the unsupervised manner. Then, this trained encoder is desired to
        use the strong backbone by downstream tasks to solve the objectives effectively.
            Therefore, the train loop here utilize the
            - trainset with one specific transform (contrastive data augmentation)
            - self.model, the ssl method to be trained.
        """

        batch_size = config['batch_size']
        model_type = config['model_name']
        current_round = kwargs['current_round']
        run_id = config['run_id']
        # Obtain the logging interval
        epochs = config['epochs']

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

        iterations_per_epoch = len(streamed_train_loader)
        # default not to perform any logging
        epoch_log_interval = epochs + 1
        batch_log_interval = iterations_per_epoch
        epoch_model_log_interval = epochs + 1
        if "epoch_log_interval" in config:
            epoch_log_interval = config['epoch_log_interval']
        if "batch_log_interval" in config:
            batch_log_interval = config['batch_log_interval']
        if "epoch_model_log_interval" in config:
            epoch_model_log_interval = config['epoch_model_log_interval']

        # Initializing the loss criterion
        loss_criterion = self.loss_criterion(self.model)

        # Initializing the optimizer
        optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        lr_schedule = optimizers.get_dynamic_lr_schedule(
            optimizer, iterations_per_epoch, streamed_train_loader)

        # Updated the lr_schedule to the latest status
        if lr_schedule_base_epoch != 0:
            for visit_epoch in range(1, lr_schedule_base_epoch + 1):
                lr_schedule.step(visit_epoch)

        train_checkpoint_saver = self.get_checkppint_saver()

        # Before the training, we expect to save the initial
        # model of this round
        initial_filename = client_get_name(model_name=model_type,
                                           client_id=self.client_id,
                                           round_n=current_round,
                                           epoch_n=0,
                                           run_id=run_id,
                                           ext="pth")
        train_checkpoint_saver.save_checkpoint(
            model_state_dict=self.model.state_dict(),
            check_points_name=[initial_filename],
            optimizer_state_dict=optimizer.state_dict(),
            lr_scheduler_state_dict=lr_schedule.state_dict(),
            epoch=lr_schedule_base_epoch,
            config_args=Config().to_dict())

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Define the container to hold the logging information
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        # Start training
        for epoch in range(1, epochs + 1):
            epoch_loss_meter.reset()
            # Use a default training loop
            for batch_id, (examples,
                           labels) in enumerate(streamed_train_loader):
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
                outputs = self.model(examples)
                loss = loss_criterion(outputs, labels)

                # Perform the backpropagation
                loss.backward()
                optimizer.step()

                # Update the loss data in the logging container
                epoch_loss_meter.update(loss.data.item())
                batch_loss_meter.update(loss.data.item())

                # Performe logging of one batch
                if batch_id % batch_log_interval == 0 or batch_id == iterations_per_epoch - 1:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            iterations_per_epoch - 1, batch_loss_meter.avg)
                    else:
                        logging.info(
                            "   [Client #%d] Contrastive Pre-train Epoch: \
                            [%d/%d][%d/%d]\tLoss: %.6f", self.client_id, epoch,
                            epochs, batch_id, iterations_per_epoch - 1,
                            batch_loss_meter.avg)

            # Update the learning rate
            # based on the
            lr_schedule.step(lr_schedule_base_epoch + epoch)

            # Performe logging of epochs
            if (epoch - 1) % epoch_log_interval == 0 or epoch == epochs:
                logging.info(
                    "[Client #%d] Contrastive Pre-train Epoch: [%d/%d]\tLoss: %.6f",
                    self.client_id, epoch, epochs, epoch_loss_meter.avg)

            if (epoch - 1) % epoch_model_log_interval == 0 or epoch == epochs:
                # the model generated during each round will be stored in the
                # checkpoints
                filename = client_get_name(model_name=model_type,
                                           client_id=self.client_id,
                                           round_n=current_round,
                                           epoch_n=epoch,
                                           run_id=run_id,
                                           ext="pth")
                train_checkpoint_saver.save_checkpoint(
                    model_state_dict=self.model.state_dict(),
                    check_points_name=[filename],
                    optimizer_state_dict=optimizer.state_dict(),
                    lr_scheduler_state_dict=lr_schedule.state_dict(),
                    epoch=epoch,
                    config_args=Config().to_dict())

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

        if 'max_concurrency' in config:
            # the final of each round, the trained model within this round
            # will be saved as model to the '/models' dir
            model_type = config['model_name']
            current_round = kwargs['current_round']
            filename = client_get_name(model_name=model_type,
                                       client_id=self.client_id,
                                       round_n=current_round,
                                       run_id=config['run_id'],
                                       ext="pth")
            # if final round, save to the model path
            if current_round == Config().trainer.rounds:
                target_dir = Config().params['model_path']
            else:
                target_dir = Config().params['checkpoint_path']

            to_save_dir = os.path.join(target_dir,
                                       "client_" + str(self.client_id))
            cpk_saver = checkpoint_saver.CheckpointsSaver(
                checkpoints_dir=to_save_dir)
            cpk_saver.save_checkpoint(
                model_state_dict=self.model.state_dict(),
                check_points_name=[filename],
                optimizer_state_dict=optimizer.state_dict(),
                lr_scheduler_state_dict=lr_schedule.state_dict(),
                epoch=epochs,
                config_args=Config().to_dict())

    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.

        Note:
            This function performs the test process of the self-supervised learning.
        Thus, it aims to measure the quanlity of the learned representation. Generally,
        the monitors, cluster algorithms, will be used to make a classification within
        the test dataset. The general pipeline is:
            trainset -> pre-trained ssl encoder -> extracted samples' features.
            extracted samples' features -> knn -> clusters.
            testset -> trained knn -> classification.
            Therefore, the monitor process here utilize the
            - trainset with general transform
            - testset with general transform
        """
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            if sampler is None:
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=config['batch_size'], shuffle=False)
                if "monitor_trainset" in kwargs:
                    monitor_train_loader = torch.utils.data.DataLoader(
                        kwargs["monitor_trainset"],
                        batch_size=config['batch_size'],
                        shuffle=False)
            # Use a testing set following the same distribution as the training set
            else:
                test_loader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    sampler=sampler.get())
                if "monitor_trainset" in kwargs:
                    monitor_train_loader = torch.utils.data.DataLoader(
                        kwargs["monitor_trainset"],
                        batch_size=config['batch_size'],
                        shuffle=False,
                        sampler=kwargs["monitor_trainset_sampler"].get())
            # Perform the monitor process to evaluate the representation
            accuracy = ssl_monitor_register.get()(
                encoder=self.model.encoder,
                monitor_data_loader=monitor_train_loader,
                test_data_loader=test_loader,
                device=self.device)

        except Exception as testing_exception:
            logging.info("Monitor Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            # first save the monitor to the results path
            result_path = Config().params['result_path']

            save_location = os.path.join(result_path,
                                         "client_" + str(self.client_id))

            current_round = kwargs['current_round']
            filename = client_get_name(client_id=self.client_id,
                                       suffix="monitor",
                                       run_id=config['run_id'],
                                       ext="csv")
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_accuracy(accuracy,
                                            round=current_round,
                                            accuracy_type="monitor_accuracy",
                                            filename=filename,
                                            location=save_location)
            # save current accuracy directly for the latter usage
            # in the test(...)
            # this is the one used in source plato,
            # do not change
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def eval_test_process(self, config, testset, sampler=None, **kwargs):
        """ The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.

        Note:
            This is second stage of evaluation in general self-supervised learning
        methods. In this stage, the learned representation will be used for downstream
        tasks, such as image classification. The pipeline is:
            task_input -> pretrained ssl_encoder -> representation -> task_solver -> task_loss.

            But in the federated learning domain, each client performs this stage on its
        local data. The main target is to train the personalized model to complete its
        own task. Thus, the task_solver mentioned above is the personalized_model.

            By the way, the upper mentioned 'pretrained ssl_encoder' is the self.model
        in the federated learning implementation. As this is the only model shared among
        clients.
        """
        personalized_model_name = Config().trainer.personalized_model_name
        current_round = kwargs['current_round']

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            assert "eval_trainset" in kwargs and "eval_trainset_sampler" in kwargs

            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['pers_batch_size'],
                shuffle=False,
                sampler=sampler.get())

            eval_train_loader = torch.utils.data.DataLoader(
                kwargs["eval_trainset"],
                batch_size=config['pers_batch_size'],
                shuffle=False,
                sampler=kwargs["eval_trainset_sampler"].get())

            # Perform the evaluation in the downstream task
            #   i.e., the client's personal local dataset
            eval_optimizer = optimizers.get_dynamic_optimizer(
                self.personalized_model, prefix="pers_")
            iterations_per_epoch = np.ceil(
                len(kwargs["eval_trainset"]) /
                Config().trainer.pers_batch_size).astype(int)

            # Initializing the learning rate schedule, if necessary
            assert 'pers_lr_schedule' in config
            lr_schedule = optimizers.get_dynamic_lr_schedule(
                optimizer=eval_optimizer,
                iterations_per_epoch=iterations_per_epoch,
                train_loader=eval_train_loader,
                prefix="pers_")

            eval_train_checkpoint_saver = self.get_checkppint_saver()

            # Before the training, we expect to save the initial
            # model of this round
            initial_filename = client_get_name(
                client_id=self.client_id,
                prefix="personalized",
                model_name=personalized_model_name,
                round_n=current_round,
                epoch_n=0,
                run_id=config['run_id'],
                ext="pth")
            eval_train_checkpoint_saver.save_checkpoint(
                model_state_dict=self.personalized_model.state_dict(),
                check_points_name=[initial_filename],
                optimizer_state_dict=eval_optimizer.state_dict(),
                lr_scheduler_state_dict=lr_schedule.state_dict(),
                epoch=0,
                config_args=Config().to_dict())

            # Initializing the loss criterion
            _eval_loss_criterion = getattr(self, "eval_loss_criterion", None)
            if callable(_eval_loss_criterion):
                eval_loss_criterion = self.eval_loss_criterion(self.model)
            else:
                eval_loss_criterion = torch.nn.CrossEntropyLoss()

            self.personalized_model.to(self.device)
            self.model.to(self.device)
            self.personalized_model.train()
            self.model.train()

            # Define the training and logging information
            # default not to perform any logging
            pers_epochs = Config().trainer.pers_epochs
            epoch_log_interval = pers_epochs + 1
            epoch_model_log_interval = pers_epochs + 1

            if "pers_epoch_log_interval" in config:
                epoch_log_interval = config['pers_epoch_log_interval']

            if "epoch_model_log_interval" in config:
                epoch_model_log_interval = config[
                    'pers_epoch_model_log_interval']

            # epoch loss tracker
            epoch_loss_meter = optimizers.AverageMeter(name='Loss')
            # encoded data
            train_encoded = list()
            train_labels = list()
            test_encoded = list()
            test_labels = list()
            # Start eval training
            # Note:
            #   To distanguish the eval training stage with the
            # previous ssl's training stage. We utilize the progress bar
            # to demonstrate the training progress details.
            global_progress = tqdm(range(1, pers_epochs + 1),
                                   desc='Evaluating')

            for epoch in global_progress:
                epoch_loss_meter.reset()

                local_progress = tqdm(eval_train_loader,
                                      desc=f'Epoch {epoch}/{pers_epochs+1}',
                                      disable=True)

                for _, (examples, labels) in enumerate(local_progress):
                    examples, labels = examples.to(self.device), labels.to(
                        self.device)
                    # Clear the previous gradient
                    eval_optimizer.zero_grad()

                    # Extract representation from the trained
                    # frozen encoder of ssl.
                    # No optimization is reuqired by this encoder.
                    with torch.no_grad():
                        feature = self.model.encoder(examples)

                    # Perfrom the training and compute the loss
                    preds = self.personalized_model(feature)
                    loss = eval_loss_criterion(preds, labels)

                    # Perfrom the optimization
                    loss.backward()
                    eval_optimizer.step()

                    # Update the epoch loss container
                    epoch_loss_meter.update(loss.data.item())

                    # save the encoded train data of current epoch
                    if epoch == pers_epochs:
                        train_encoded.append(feature)
                        train_labels.append(labels)

                    local_progress.set_postfix({
                        'lr': lr_schedule,
                        "loss": epoch_loss_meter.val,
                        'loss_avg': epoch_loss_meter.avg
                    })

                if (epoch -
                        1) % epoch_log_interval == 0 or epoch == pers_epochs:
                    logging.info(
                        "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                        self.client_id, epoch, pers_epochs,
                        epoch_loss_meter.avg)

                if (epoch - 1
                    ) % epoch_model_log_interval == 0 or epoch == pers_epochs:
                    # the model generated during each round will be stored in the
                    # checkpoints

                    filename = client_get_name(
                        client_id=self.client_id,
                        prefix="personalized",
                        model_name=personalized_model_name,
                        round_n=current_round,
                        epoch_n=epoch,
                        run_id=config['run_id'],
                        ext="pth")

                    eval_train_checkpoint_saver.save_checkpoint(
                        model_state_dict=self.personalized_model.state_dict(),
                        check_points_name=[filename],
                        optimizer_state_dict=eval_optimizer.state_dict(),
                        lr_scheduler_state_dict=lr_schedule.state_dict(),
                        epoch=epoch,
                        config_args=Config().to_dict())

                lr_schedule.step()

            # Define the test phase of the eval stage
            acc_meter = optimizers.AverageMeter(name='Accuracy')

            self.personalized_model.eval()
            self.model.eval()
            correct = 0

            acc_meter.reset()
            for _, (examples, labels) in enumerate(test_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                with torch.no_grad():
                    feature = self.model.encoder(examples)
                    preds = self.personalized_model(feature).argmax(dim=1)
                    correct = (preds == labels).sum().item()
                    acc_meter.update(correct / preds.shape[0])
                    test_encoded.append(feature)
                    test_labels.append(labels)

            accuracy = acc_meter.avg
        except Exception as testing_exception:
            logging.info("Evaluation Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        # save the personalized model for current round
        # to the model dir of this client
        if 'max_concurrency' in config:

            current_round = kwargs['current_round']
            if current_round == Config().trainer.rounds:
                target_dir = Config().params['model_path']
            else:
                target_dir = Config().params['checkpoint_path']
            personalized_model_name = Config().trainer.personalized_model_name
            save_location = os.path.join(target_dir,
                                         "client_" + str(self.client_id))
            filename = client_get_name(client_id=self.client_id,
                                       model_name=personalized_model_name,
                                       round_n=current_round,
                                       run_id=config['run_id'],
                                       prefix="personalized",
                                       ext="pth")
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename,
                                         location=save_location)

        # save the accuracy of the client
        if 'max_concurrency' in config:

            # save the encoded data to the results dir
            result_path = Config().params['result_path']
            save_location = os.path.join(result_path,
                                         "client_" + str(self.client_id))
            train_filename = client_get_name(client_id=self.client_id,
                                             round_n=current_round,
                                             epoch_n=epoch,
                                             run_id=config['run_id'],
                                             suffix="trainEncoded",
                                             ext="npy")
            test_filename = train_filename.replace("trainEncoded",
                                                   "testEncoded")
            self.save_encoded_data(encoded_data=train_encoded,
                                   data_labels=train_labels,
                                   filename=train_filename,
                                   location=save_location)

            self.save_encoded_data(encoded_data=test_encoded,
                                   data_labels=test_labels,
                                   filename=test_filename,
                                   location=save_location)
        # if we do not want to keep the state of the personalized model
        # at the end of this round,
        # we need to load the initial model
        if not (hasattr(Config().trainer, "do_maintain_per_state")
                and Config().trainer.do_maintain_per_state):
            location = eval_train_checkpoint_saver.checkpoints_dir
            load_from_path = os.path.join(location, initial_filename)

            self.personalized_model.load_state_dict(
                torch.load(load_from_path)["model"], strict=True)

            logging.info(
                "[Client #%d] recall the initial personalized model of round %d",
                self.client_id, current_round)

        # save the accuracy of the client
        if 'max_concurrency' in config:
            # save the personaliation accuracy to the results dir
            result_path = Config().params['result_path']

            save_location = os.path.join(result_path,
                                         "client_" + str(self.client_id))

            current_round = kwargs['current_round']
            filename = client_get_name(client_id=self.client_id,
                                       run_id=config['run_id'],
                                       suffix="personalization",
                                       ext="csv")
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_accuracy(
                accuracy,
                round=current_round,
                accuracy_type="personalization_accuracy",
                filename=filename,
                location=save_location)

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config['personalized_model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)

        else:
            return accuracy

    def eval_test(self, testset, sampler=None, **kwargs) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.

        Note:
            It performs 'eval_test_process' that is responsible for evaluating
        the learned representation of ssl.
            Under the terminologies of federated learning, this part is to train
        the personalized model.
            However, we still call it 'eval_test' just to align the learning stage
        with that in self-supervised learning (ssl), i.e., evaluation stage.

        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.eval_test_process,
                              args=(config, testset, sampler),
                              kwargs=kwargs)
            proc.start()
            proc.join()

            accuracy = -1
            try:
                model_name = Config().trainer.personalized_model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Evaluation testing on client #{self.client_id} failed."
                ) from error

            self.pause_training()
        else:
            accuracy = self.eval_test_process(config, testset, **kwargs)

        return accuracy
