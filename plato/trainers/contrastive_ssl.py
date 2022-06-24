"""
Implement the trainer for contrastive self-supervised learning method.

"""

import os
import logging
import time
import multiprocessing as mp
import copy
from attr import has

import numpy as np
import torch

from tqdm import tqdm
import pandas as pd

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers
from plato.utils import data_loaders_wrapper

from plato.models import ssl_monitor_register

from plato.utils import ssl_losses


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

    def save_encoded_data(self, data, filename=None, location=None):
        """ Save the encoded data (np.narray). """
        # process the arguments to obtain the to save path
        to_save_path = self.process_save_path(filename,
                                              location,
                                              work_model_name="model_name",
                                              desired_extenstion=".npy")

        np.save(to_save_path, data, allow_pickle=True)
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

        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler,
                                             cut_layer)
        else:
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
                sampler=unlabeled_sampler)

        # wrap the multiple loaders into one sequence loader
        streamed_train_loader = data_loaders_wrapper.StreamBatchesLoader(
            [train_loader, unlabeled_loader])

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        iterations_per_epoch += np.ceil(len(unlabeled_trainset) /
                                        batch_size).astype(int)

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        loss_criterion = self.loss_criterion(self.model)

        # Initializing the optimizer
        optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if 'lr_schedule' in config:
            lr_schedule = optimizers.get_dynamic_lr_schedule(
                optimizer, iterations_per_epoch, streamed_train_loader)
        else:
            lr_schedule = None

        # Obtain the logging interval
        epochs = config['epochs']
        # do not save the model during epoch training of each round
        epoch_model_log_interval = epochs
        epoch_log_interval = config['epoch_log_interval']
        batch_log_interval = config['batch_log_interval']

        if "epoch_model_log_interval" in config:
            epoch_model_log_interval = config['epoch_model_log_interval']

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

                # Performe logging of batches
                if batch_id % batch_log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(streamed_train_loader), batch_loss_meter.avg)
                    else:
                        logging.info(
                            "   [Client #%d] Contrastive Pre-train Epoch: \
                            [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(streamed_train_loader), batch_loss_meter.avg)

            # Performe logging of epochs
            if (epoch - 1) % epoch_log_interval == 0:
                logging.info(
                    "[Client #%d] Contrastive Pre-train Epoch: [%d/%d]\tLoss: %.6f",
                    self.client_id, epoch, epochs, epoch_loss_meter.avg)

            if (epoch - 1) % epoch_model_log_interval == 0:
                # the model generated during each round will be stored in the
                # checkpoints
                model_type = config['model_name']
                current_round = kwargs['current_round']
                filename = f"{model_type}_client{self.client_id}_round{current_round}_epoch{epoch}_runid{config['run_id']}.pth"
                target_dir = Config().params['checkpoint_path']
                to_save_dir = os.path.join(target_dir,
                                           "client_" + str(self.client_id))
                os.makedirs(to_save_dir, exist_ok=True)

                torch.save(self.model.state_dict(),
                           os.path.join(to_save_dir, filename))

            # Update the learning rate
            if lr_schedule is not None:
                lr_schedule.step()

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
            self.model.cpu()
            model_type = config['model_name']
            current_round = kwargs['current_round']
            filename = f"{model_type}_client{self.client_id}_round{current_round}_runid{config['run_id']}.pth"
            # if final round, save to the model path
            if current_round == Config().trainer.rounds:
                target_dir = Config().params['model_path']
            else:
                target_dir = Config().params['checkpoint_path']

            to_save_dir = os.path.join(target_dir,
                                       "client_" + str(self.client_id))

            self.save_model(filename, location=to_save_dir)

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
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False)
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
            filename = f"client_{self.client_id}_monitor.csv"

            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_accuracy(accuracy,
                                            round=current_round,
                                            accuracy_type="monitor_accuracy",
                                            filename=filename,
                                            location=save_location)
            # save current accuracy directly for the latter usage
            # in the test(...)
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
        if hasattr(Config().trainer, "do_maintain_per_state") and Config(
        ).trainer.do_maintain_per_state:
            personalized_model = copy.deepcopy(self.personalized_model)
        else:
            personalized_model = self.personalized_model

        personalized_model.to(self.device)
        self.model.to(self.device)
        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            custom_test = getattr(self, "eval_test_model", None)

            if callable(custom_test):
                accuracy = self.eval_test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['pers_batch_size'],
                        shuffle=False)
                    if "eval_trainset" in kwargs:
                        eval_train_loader = torch.utils.data.DataLoader(
                            kwargs["eval_trainset"],
                            batch_size=config['pers_batch_size'],
                            shuffle=False)
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['pers_batch_size'],
                        shuffle=False,
                        sampler=sampler.get())
                    if "eval_trainset" in kwargs:
                        eval_train_loader = torch.utils.data.DataLoader(
                            kwargs["eval_trainset"],
                            batch_size=config['pers_batch_size'],
                            shuffle=False,
                            sampler=kwargs["eval_trainset_sampler"].get())

                # Perform the evaluation in the downstream task
                #   i.e., the client's personal local dataset
                eval_optimizer = optimizers.get_dynamic_optimizer(
                    personalized_model, prefix="pers_")
                iterations_per_epoch = np.ceil(
                    len(kwargs["eval_trainset"]) /
                    Config().trainer.pers_batch_size).astype(int)

                # Initializing the learning rate schedule, if necessary
                if 'pers_lr_schedule' in config:
                    lr_schedule = optimizers.get_dynamic_lr_schedule(
                        optimizer=eval_optimizer,
                        iterations_per_epoch=iterations_per_epoch,
                        train_loader=eval_train_loader,
                        prefix="pers_")
                else:
                    lr_schedule = None

                # Initializing the loss criterion
                _eval_loss_criterion = getattr(self, "eval_loss_criterion",
                                               None)
                if callable(_eval_loss_criterion):
                    eval_loss_criterion = self.eval_loss_criterion(self.model)
                else:
                    eval_loss_criterion = torch.nn.CrossEntropyLoss()

                self.model.eval()
                personalized_model.train()

                # Define the training and logging information
                epoch_log_interval = config['pers_epoch_log_interval']
                num_eval_train_epochs = Config().trainer.pers_epochs
                epoch_loss_meter = optimizers.AverageMeter(name='Loss')

                # Start eval training
                # Note:
                #   To distanguish the eval training stage with the
                # previous ssl's training stage. We utilize the progress bar
                # to demonstrate the training progress details.
                global_progress = tqdm(range(1, num_eval_train_epochs + 1),
                                       desc='Evaluating')
                train_data_encoded = list()
                train_data_labels = list()
                for epoch in global_progress:
                    epoch_loss_meter.reset()
                    local_progress = tqdm(
                        eval_train_loader,
                        desc=f'Epoch {epoch}/{num_eval_train_epochs+1}',
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
                        preds = personalized_model(feature)
                        loss = eval_loss_criterion(preds, labels)

                        # Perfrom the optimization
                        loss.backward()
                        eval_optimizer.step()

                        # Update the epoch loss container
                        epoch_loss_meter.update(loss.data.item())
                        train_data_encoded.append(feature)
                        train_data_labels.append(labels)

                        if lr_schedule is not None:
                            lr_schedule = lr_schedule.step()

                        local_progress.set_postfix({
                            'lr':
                            lr_schedule,
                            "loss":
                            epoch_loss_meter.val,
                            'loss_avg':
                            epoch_loss_meter.avg
                        })

                    if (epoch - 1) % epoch_log_interval == 0:
                        logging.info(
                            "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, num_eval_train_epochs,
                            epoch_loss_meter.avg)

                # Define the test phase of the eval stage
                acc_meter = optimizers.AverageMeter(name='Accuracy')

                personalized_model.eval()
                correct = 0
                test_data_encoded = list()
                test_data_labels = list()
                acc_meter.reset()
                for _, (examples, labels) in enumerate(test_loader):
                    examples, labels = examples.to(self.device), labels.to(
                        self.device)
                    with torch.no_grad():
                        feature = self.model.encoder(examples)
                        preds = personalized_model(feature).argmax(dim=1)
                        correct = (preds == labels).sum().item()
                        acc_meter.update(correct / preds.shape[0])
                        test_data_encoded.append(feature)
                        test_data_labels.append(labels)

                accuracy = acc_meter.avg
        except Exception as testing_exception:
            logging.info("Evaluation Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        # save the personalized model for current round
        # to the dir of this client
        if 'max_concurrency' in config:
            personalized_model.cpu()

            current_round = kwargs['current_round']
            if current_round == Config().trainer.rounds:
                target_dir = Config().params['model_path']
            else:
                target_dir = Config().params['checkpoint_path']
            personalized_model_name = Config().trainer.personalized_model_name
            save_location = os.path.join(target_dir,
                                         "client_" + str(self.client_id))
            filename = f"personalized_({personalized_model_name})_client{self.client_id}_round{current_round}_.pth"

            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename,
                                         location=save_location)

        # save the accuracy of the client
        if 'max_concurrency' in config:
            # save the personaliation accuracy to the results dir
            result_path = Config().params['result_path']

            save_location = os.path.join(result_path,
                                         "client_" + str(self.client_id))

            current_round = kwargs['current_round']
            filename = f"client_{self.client_id}_personalization.csv"
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

        # save the encoded data to the clients
        if 'max_concurrency' in config:
            # save the encoded data to the results dir
            result_path = Config().params['result_path']
            save_location = os.path.join(result_path,
                                         "client_" + str(self.client_id))
            current_round = kwargs['current_round']
            train_encoded_filename = f"Round_{current_round}_train_encoded.npy"
            train_label_filename = f"Round_{current_round}_train_label.npy"
            test_encoded_filename = f"Round_{current_round}_test_encoded.npy"
            test_label_filename = f"Round_{current_round}_test_label.npy"

            # reduce the to save data size if the recorded data is too large
            # to be saved
            if len(train_data_encoded) > 5000:
                interval = len(train_data_encoded) // 5000
                select_index = range(0, train_data_encoded, interval)
                train_data_encoded = train_data_encoded[select_index]
                train_data_labels = train_data_labels[select_index]
            train_data_encoded = torch.cat(train_data_encoded, axis=0)
            train_data_labels = torch.cat(train_data_labels)
            test_data_encoded = torch.cat(test_data_encoded, axis=0)
            test_data_labels = torch.cat(test_data_labels)

            self.save_encoded_data(
                data=train_data_encoded.detach().to("cpu").numpy(),
                filename=train_encoded_filename,
                location=save_location)
            self.save_encoded_data(
                data=train_data_labels.detach().to("cpu").numpy(),
                filename=train_label_filename,
                location=save_location)
            self.save_encoded_data(
                data=test_data_encoded.detach().to("cpu").numpy(),
                filename=test_encoded_filename,
                location=save_location)
            self.save_encoded_data(
                data=test_data_labels.detach().to("cpu").numpy(),
                filename=test_label_filename,
                location=save_location)
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
