"""
Implement the trainer for contrastive self-supervised learning method.

    Note:
        This is the training stage of self-supervised learning (ssl). It is responsible
    for performing the contrastive learning process based on the trainset to train
    a encoder in the unsupervised manner. Then, this trained encoder is desired to
    use the strong backbone by downstream tasks to solve the objectives effectively.
        Therefore, the train loop here utilize the
        - trainset with one specific transform (contrastive data augmentation)
        - self.model, the ssl method to be trained.
"""

import os
import logging

import multiprocessing as mp
import warnings

warnings.simplefilter('ignore')

import numpy as np
import torch

from tqdm import tqdm

from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers
from plato.utils.checkpoint_operator import perform_client_checkpoint_saving
from plato.utils.checkpoint_operator import get_client_checkpoint_operator
from plato.utils.arrange_saving_name import get_format_name
from plato.models import ssl_monitor_register

from plato.utils import ssl_losses
from plato.utils.arrange_saving_name import get_format_name


class Trainer(pers_basic.Trainer):
    """ A federated learning trainer for contrastive self-supervised methods. """

    def __init__(self, model=None):
        super().__init__(model)

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
            filename = get_format_name(client_id=self.client_id,
                                       suffix="monitor",
                                       run_id=None,
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

    def perform_evaluation_op(
        self,
        test_loader,
    ):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
        test_encoded = list()
        test_labels = list()
        self.personalized_model.eval()
        self.model.eval()
        correct = 0

        acc_meter.reset()
        for _, (examples, labels) in enumerate(test_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                feature = self.model.encoder(examples)
                preds = self.personalized_model(feature).argmax(dim=1)
                correct = (preds == labels).sum().item()
                acc_meter.update(correct / preds.shape[0], labels.size(0))
                test_encoded.append(feature)
                test_labels.append(labels)

        accuracy = acc_meter.avg

        return accuracy, test_encoded, test_labels

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

            # Before the training, we expect to save the initial
            # model of this round
            initial_filename = perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=personalized_model_name,
                model_state_dict=self.personalized_model.state_dict(),
                config=config,
                current_round=current_round,
                optimizer_state_dict=eval_optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=0,
                base_epoch=0,
                prefix="personalized")

            accuracy, _, _ = self.perform_test_op(test_loader)
            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(
                accuracy=accuracy,
                current_round=kwargs['current_round'],
                epoch=0,
                run_id=None)

            # Initializing the loss criterion
            _eval_loss_criterion = getattr(self, "pers_loss_criterion", None)
            if callable(_eval_loss_criterion):
                eval_loss_criterion = self.pers_loss_criterion(self.model)
            else:
                eval_loss_criterion = torch.nn.CrossEntropyLoss()

            self.personalized_model.to(self.device)
            self.model.to(self.device)

            # Define the training and logging information
            # default not to perform any logging
            pers_epochs = Config().trainer.pers_epochs
            epoch_log_interval = pers_epochs + 1
            epoch_model_log_interval = pers_epochs + 1

            if "pers_epoch_log_interval" in config:
                epoch_log_interval = config['pers_epoch_log_interval']

            if "pers_epoch_model_log_interval" in config:
                epoch_model_log_interval = config[
                    'pers_epoch_model_log_interval']

            # epoch loss tracker
            epoch_loss_meter = optimizers.AverageMeter(name='Loss')
            # encoded data
            train_encoded = list()
            train_labels = list()
            # Start eval training
            # Note:
            #   To distanguish the eval training stage with the
            # previous ssl's training stage. We utilize the progress bar
            # to demonstrate the training progress details.
            global_progress = tqdm(range(1, pers_epochs + 1),
                                   desc='Evaluating')

            for epoch in global_progress:
                epoch_loss_meter.reset()
                self.personalized_model.train()
                self.model.eval()
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
                    epoch_loss_meter.update(loss.data.item(), labels.size(0))

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

                    accuracy, _, _ = self.perform_test_op(test_loader)
                    # save the personaliation accuracy to the results dir
                    self.checkpoint_personalized_accuracy(
                        accuracy=accuracy,
                        current_round=kwargs['current_round'],
                        epoch=epoch,
                        run_id=None)

                if (epoch - 1
                    ) % epoch_model_log_interval == 0 or epoch == pers_epochs:
                    # the model generated during each round will be stored in the
                    # checkpoints
                    perform_client_checkpoint_saving(
                        client_id=self.client_id,
                        model_name=personalized_model_name,
                        model_state_dict=self.personalized_model.state_dict(),
                        config=config,
                        current_round=current_round,
                        optimizer_state_dict=eval_optimizer.state_dict(),
                        lr_schedule_state_dict=lr_schedule.state_dict(),
                        present_epoch=epoch,
                        base_epoch=epoch,
                        prefix="personalized")

                lr_schedule.step()

            accuracy, test_encoded, test_labels = self.perform_test_op(
                test_loader)

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
            filename = get_format_name(client_id=self.client_id,
                                       model_name=personalized_model_name,
                                       round_n=current_round,
                                       run_id=None,
                                       prefix="personalized",
                                       ext="pth")
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename,
                                         location=save_location)

        # save the accuracy of the client
        if 'max_concurrency' in config:

            self.checkpoint_encoded_samples(encoded_samples=train_encoded,
                                            encoded_labels=train_labels,
                                            current_round=current_round,
                                            epoch=epoch,
                                            run_id=None,
                                            encoded_type="trainEncoded")
            self.checkpoint_encoded_samples(encoded_samples=test_encoded,
                                            encoded_labels=test_labels,
                                            current_round=current_round,
                                            epoch=epoch,
                                            run_id=None,
                                            encoded_type="testEncoded")

        # save the accuracy of the client
        if 'max_concurrency' in config:

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