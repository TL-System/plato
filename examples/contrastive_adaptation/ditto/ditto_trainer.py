"""
A personalized federated learning trainer using Ditto.

"""

import os
import copy
import time
import logging
import warnings
import collections

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
import numpy as np

from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers
from plato.utils.arrange_saving_name import get_format_name
from plato.utils.checkpoint_operator import perform_client_checkpoint_saving
from plato.utils.checkpoint_operator import perform_client_checkpoint_loading

from plato.utils import data_loaders_wrapper


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def ditto_solver(
        self,
        defined_model,
        baseline_ditto_weights,
        train_loader,
        config,
    ):
        """ Perform the ditto solver to train the personalized model.

        The personalization model of Dittois trained with the Ditto solver, which
        is denoted as:
            v_k = v_k - η(∇F_k(v_k) + λ(v_k - w^t))

        This can be witnessed in the FedRep's implementation of Ditto.
            https://github.com/lgcollins/FedRep

        """
        pers_epochs = config['pers_epochs']
        lamda = config['lamda']

        epoch_log_interval = pers_epochs + 1
        if "pers_epoch_log_interval" in config:
            epoch_log_interval = config['pers_epoch_log_interval']

        # Perform the personalization
        #   i.e., the client's personal local dataset
        pers_optimizer = optimizers.get_dynamic_optimizer(defined_model,
                                                          prefix="pers_")
        iterations_per_epoch = len(train_loader)

        # Initializing the learning rate schedule, if necessary
        assert 'pers_lr_schedule' in config
        lr_schedule = optimizers.get_dynamic_lr_schedule(
            optimizer=pers_optimizer,
            iterations_per_epoch=iterations_per_epoch,
            train_loader=train_loader,
            prefix="pers_")

        # Initializing the loss criterion
        _pers_loss_criterion = getattr(self, "pers_loss_criterion", None)
        if callable(_pers_loss_criterion):
            pers_loss_criterion = self.pers_loss_criterion(defined_model)
        else:
            pers_loss_criterion = torch.nn.CrossEntropyLoss()

        defined_model.train()
        defined_model.to(self.device)

        # Start personalization training with the Ditto solver
        # Note:
        #   To distanguish the eval training stage with the
        # previous ssl's training stage. We utilize the progress bar
        # to demonstrate the training progress details.
        global_progress = tqdm(range(1, pers_epochs + 1), desc='Ditto Solver')
        # epoch loss tracker
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')

        for epoch in global_progress:
            epoch_loss_meter.reset()
            local_progress = tqdm(train_loader,
                                  desc=f'Epoch {epoch}/{pers_epochs+1}',
                                  disable=True)

            for _, (examples, labels) in enumerate(local_progress):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                # backup the params of defined model before optimization
                # this is the v_k in the Algorithm. 1
                v_initial = copy.deepcopy(defined_model.state_dict())
                # Clear the previous gradient
                pers_optimizer.zero_grad()

                ## 1.- Compute the ∇F_k(v_k), thus to compute the first term
                #   of the equation in the Algorithm. 1.
                # i.e., v_k − η∇F_k(v_k)
                # This can be achieved by the general optimization step.
                # Perfrom the training and compute the loss
                preds = defined_model(examples)
                loss = pers_loss_criterion(preds, labels)

                # Perfrom the optimization
                loss.backward()
                pers_optimizer.step()

                ## 2.- Compute the ηλ(v_k − w^t), which is the second term of
                #   the corresponding equation in Algorithm. 1.
                w_net = copy.deepcopy(defined_model.state_dict())
                lr = lr_schedule.get_lr()[0]
                for key in w_net.keys():
                    w_net[key] = w_net[key] - lr * lamda * (
                        v_initial[key] - baseline_ditto_weights[key])
                defined_model.load_state_dict(w_net)

                # Update the epoch loss container
                epoch_loss_meter.update(loss.data.item(), labels.size(0))

            if (epoch - 1) % epoch_log_interval == 0 or epoch == pers_epochs:
                logging.info(
                    "[Client #%d] With Ditto solver, Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                    self.client_id, epoch, pers_epochs, epoch_loss_meter.avg)
            lr_schedule.step()

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

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
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

        # backup the unoptimized global model
        # this is used as the baseline ditto weights in the Ditto solver
        initial_model_params = copy.deepcopy(self.model.state_dict())

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

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

        ## Then, we perform the personalization process with the Ditto solver.
        self.ditto_solver(
            defined_model=self.personalized_model,
            baseline_ditto_weights=initial_model_params,
            train_loader=streamed_train_loader,
            config=config,
        )

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

            if current_round >= config['rounds']:
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

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
        defined_model.eval()
        defined_model.to(self.device)
        correct = 0

        encoded_samples = list()
        loaded_labels = list()

        acc_meter.reset()
        for _, (examples, labels) in enumerate(to_eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                # preds = self.personalized_model(examples).argmax(dim=1)

                features = defined_model.encoder(examples)
                preds = defined_model.clf_fc(features).argmax(dim=1)

                correct = (preds == labels).sum().item()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                encoded_samples.append(features)
                loaded_labels.append(labels)

        accuracy = acc_meter.avg

        test_outputs = {
            "accuracy": accuracy,
            "encoded_samples": encoded_samples,
            "loaded_labels": loaded_labels
        }

        return test_outputs

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
            By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        current_round = config['current_round']

        eval_outputs = self.perform_evaluation_op(data_loader, defined_model)

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=eval_outputs["accuracy"],
            current_round=current_round,
            epoch=epoch,
            run_id=None)

        self.checkpoint_encoded_samples(
            encoded_samples=eval_outputs['encoded_samples'],
            encoded_labels=eval_outputs['loaded_labels'],
            current_round=current_round,
            epoch=epoch,
            run_id=None,
            encoded_type="testEncoded")

        return eval_outputs, None

    def pers_train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

            In the Ditto method, there is no additional personalization stage, also
            presented by the Ditto implementation in FedRep and FedBABU.

            Therefore, in the personalization stage of our framework, we directly
            obtain the performance of self.personalized model.

            The reason why Ditto does not need to perform the additional training
            for personalization is that the personalized model training works with
            the global model training together, as presented by the self.train_model(*)
            function.
            Therefore, we only obtain the encoded data and the performance of the
            trained personalized model.
        """

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1
        current_round = kwargs['current_round']
        config['current_round'] = current_round
        personalized_model_name = config['personalized_model_name']
        try:
            assert "testset" in kwargs and "testset_sampler" in kwargs
            testset = kwargs['testset']
            testset_sampler = kwargs['testset_sampler']
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['pers_batch_size'],
                shuffle=False,
                sampler=testset_sampler.get())

            eval_outputs, _ = self.on_start_pers_train(
                defined_model=self.personalized_model,
                model_name=personalized_model_name,
                data_loader=test_loader,
                epoch=0,
                global_epoch=0,
                config=config,
                optimizer=None,
                lr_schedule=None,
            )
            accuracy = eval_outputs["accuracy"]

        except Exception as testing_exception:
            logging.info("Personalization Learning on client #%d failed.",
                         self.client_id)
            raise testing_exception

        if 'max_concurrency' in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config['personalized_model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)

        else:
            return accuracy