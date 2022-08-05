"""
A personalized federated learning trainer using APFL.

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
    """A personalized federated learning trainer using the APFL algorithm."""

    def obtain_encoded_data(self, defined_model, pers_train_loader,
                            test_loader):
        # encoded data
        train_encoded = list()
        train_labels = list()
        test_outputs = {}
        for _, (examples, labels) in enumerate(pers_train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            features = defined_model.encoder(examples)
            train_encoded.append(features)
            train_labels.append(labels)
        test_outputs = self.perform_test_op(test_loader, defined_model)

        return train_encoded, train_labels, test_outputs

    def load_alpha(self, current_round, epoch, run_id=None):

        filename, cpk_oper = perform_client_checkpoint_loading(
            client_id=self.client_id,
            model_name=Config().trainer.model_name,
            current_round=current_round - 1,
            run_id=run_id,
            epoch=epoch,
            prefix=None,
            anchor_metric="round",
            mask_anchors=["epoch", "personalized"],
            use_latest=True)
        if filename is None:
            logging.info(
                f"No updated alpha existed, thus use the initial alpha.")
            return None
        else:
            alpha = cpk_oper.load_checkpoint(filename)["args"]["alpha"]
            logging.info(
                f"[Client #{self.client_id}] loaded alpha {alpha} from {filename}."
            )
            return alpha

    def alpha_update(self, defined_model, personalized_model, alpha, eta):
        """ Update the alpha based on the Eq. 10 of the paper.

            The implementation of this alpha update comes from the
            APFL code of:
             https://github.com/MLOPTPSU/FedTorch/blob/main/main.py

            The only concern is that
                why 'grad_alpha' needs to be computed as:
                    grad_alpha = grad_alpha + 0.02 * alpha
        """
        grad_alpha = 0
        # perform the second term of Eq. 10
        for l_params, p_params in zip(defined_model.parameters(),
                                      personalized_model.parameters()):
            dif = p_params.data - l_params.data
            grad = alpha * p_params.grad.data + (1 -
                                                 alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * alpha

        alpha_n = alpha - eta * grad_alpha
        alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)

        return alpha_n

    def train_one_epoch(self, config, epoch, defined_model, optimizer,
                        pers_optimizer, loss_criterion, train_data_loader,
                        epoch_loss_meter, batch_loss_meter):
        defined_model.train()
        epochs = config['epochs']

        # load the alpha of the APFL method
        alpha = config['alpha']

        iterations_per_epoch = len(train_data_loader)
        # default not to perform any logging
        epoch_log_interval = epochs + 1
        batch_log_interval = iterations_per_epoch

        if "epoch_log_interval" in config:
            epoch_log_interval = config['epoch_log_interval']
        if "batch_log_interval" in config:
            batch_log_interval = config['batch_log_interval']

        epoch_pers_loss_meter = optimizers.AverageMeter(name='PersLoss')
        batch_pers_loss_meter = optimizers.AverageMeter(name='PersLoss')

        epoch_loss_meter.reset()
        epoch_pers_loss_meter.reset()
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

            ## 1.- Perform the general training on the model
            # Reset and clear previous data
            batch_loss_meter.reset()
            batch_pers_loss_meter.reset()
            optimizer.zero_grad()

            # backup the params of the global model, i.e., w_i in Algo1
            # thus, backup_ws presents the parameters of w^(t-1)_i
            backup_ws = copy.deepcopy(defined_model.state_dict())

            # Forward the model and compute the loss
            outputs = defined_model(examples)
            loss = loss_criterion(outputs, labels)

            # Perform the backpropagation
            loss.backward()
            optimizer.step()
            optimized_ws = copy.deepcopy(defined_model.state_dict())

            ## 2.- Perform the local model training
            # i.e., v_i(t) computation in Algo1
            # wt = copy.deepcopy(defined_model.state_dict())
            # defined_model.load_state_dict(w_loc_new)

            # Forward the model and compute the loss
            # This is to achieve the \overline{v}_i
            # \overline{v}^(t-1)_i = alpha v^(t-1)_i + (1-alpha) w^(t-1)_i
            # recovery the w^(t-1)
            defined_model.load_state_dict(backup_ws)

            outputs1 = defined_model(examples)
            outputs2 = self.personalized_model(examples)
            outputs = alpha * outputs1 + (1 - alpha) * outputs2
            pers_loss = loss_criterion(outputs, labels)

            pers_optimizer.zero_grad()
            pers_loss.backward()
            pers_optimizer.step()

            # finally, recovery the optimized parameters of w^t_i
            defined_model.load_state_dict(optimized_ws)

            # Update the loss data in the logging container
            epoch_loss_meter.update(loss.data.item(), labels.size(0))
            batch_loss_meter.update(loss.data.item(), labels.size(0))
            epoch_pers_loss_meter.update(pers_loss.data.item(), labels.size(0))
            batch_pers_loss_meter.update(pers_loss.data.item(), labels.size(0))

            # Performe logging of one batch
            if batch_id % batch_log_interval == 0 or batch_id == iterations_per_epoch - 1:
                logging.info(
                    "   [Client #%d] Training Epoch: \
                    [%d/%d][%d/%d]\tLoss: %.6f, PersLoss: %.6f",
                    self.client_id, epoch, epochs, batch_id,
                    iterations_per_epoch - 1, batch_loss_meter.avg,
                    batch_pers_loss_meter.avg)

        # Performe logging of epochs
        if (epoch - 1) % epoch_log_interval == 0 or epoch == epochs:
            logging.info(
                "[Client #%d] Training Epoch: [%d/%d]\tLoss: %.6f, PersLoss: %.6f",
                self.client_id, epoch, epochs, epoch_loss_meter.avg,
                epoch_pers_loss_meter.avg)

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

        batch_size = config['batch_size']
        model_type = config['model_name']
        current_round = kwargs['current_round']
        run_id = config['run_id']
        epochs = config['epochs']

        # obtained the saved alpha in the previous round
        initial_alpha = config['alpha']
        is_adaptive_alpha = config['is_adaptive_alpha']
        loaded_alpha = self.load_alpha(current_round=current_round,
                                       epoch=None,
                                       run_id=None)
        alpha = initial_alpha if loaded_alpha is None else loaded_alpha
        config['alpha'] = alpha

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
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        optimizer = optimizers.get_dynamic_optimizer(self.model)

        # Initializing the optimizer for personalized model
        pers_optimizer = optimizers.get_dynamic_optimizer(
            self.personalized_model, prefix="pers_")

        # Initializing the learning rate schedule, if necessary
        pers_lr_schedule, _ = self.prepare_train_lr(pers_optimizer,
                                                    streamed_train_loader,
                                                    config, current_round)

        # Initializing the learning rate schedule, if necessary
        lr_schedule, lr_schedule_base_epoch = self.prepare_train_lr(
            optimizer, streamed_train_loader, config, current_round)

        logging.info(
            f"With {lr_schedule}, we get lr={lr_schedule.get_lr()} under the global epoch {lr_schedule_base_epoch}"
        )

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.personalized_model.to(self.device)
        # Define the container to hold the logging information
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        # Start training
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.personalized_model.train()
            self.train_one_epoch(config,
                                 epoch,
                                 defined_model=self.model,
                                 optimizer=optimizer,
                                 pers_optimizer=pers_optimizer,
                                 loss_criterion=loss_criterion,
                                 train_data_loader=streamed_train_loader,
                                 epoch_loss_meter=epoch_loss_meter,
                                 batch_loss_meter=batch_loss_meter)

            # Update the learning rate
            # based on the base epoch
            lr_schedule.step()
            pers_lr_schedule.step()

            # update alpha based on the Eq. 10 of the paper.
            if is_adaptive_alpha and epoch == 1:
                #0.1/np.sqrt(1+args.local_index))
                lr = lr_schedule.get_lr()[0]
                previous_alpha = alpha
                alpha = self.alpha_update(self.model, self.personalized_model,
                                          alpha, lr)
                config['alpha'] = alpha
                logging.info(
                    "[Client #%d] in round#%d Update alpha from %.6f to %.6f.",
                    self.client_id, current_round, previous_alpha, alpha)

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
                kwargs=kwargs,
                optimizer_state_dict=optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=None,
                base_epoch=lr_schedule_base_epoch + epochs)

            current_round = kwargs['current_round']
            if current_round >= Config().trainer.rounds:
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

    def perform_test_op(self, test_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
        defined_model.eval()
        defined_model.to(self.device)
        correct = 0

        test_encoded = list()
        test_labels = list()

        acc_meter.reset()
        for _, (examples, labels) in enumerate(test_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                # preds = self.personalized_model(examples).argmax(dim=1)

                features = defined_model.encoder(examples)
                preds = defined_model.clf_fc(features).argmax(dim=1)

                correct = (preds == labels).sum().item()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                test_encoded.append(features)
                test_labels.append(labels)

        accuracy = acc_meter.avg

        test_outputs = {
            "accuracy": accuracy,
            "test_encoded": test_encoded,
            "test_labels": test_labels
        }

        return test_outputs

    def pers_train_model(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

            In the APFL method, there is no additional personalization stage, also
            presented by the APFL implementation in FedTorch[1].

            Therefore, in the personalization stage of our framework, we directly
            obtain the performance of self.personalized model.

            [1]. https://github.com/MLOPTPSU/FedTorch/blob/main/main.py

            The reason why APFL does not need to perform the additional training
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
            # also record the encoded data in the first epoch

            train_encoded, train_labels, test_outputs = self.obtain_encoded_data(
                self.personalized_model, pers_train_loader, test_loader)

            self.checkpoint_encoded_samples(encoded_samples=train_encoded,
                                            encoded_labels=train_labels,
                                            current_round=current_round,
                                            epoch=None,
                                            run_id=None,
                                            encoded_type="trainEncoded")
            self.checkpoint_encoded_samples(
                encoded_samples=test_outputs["test_encoded"],
                encoded_labels=test_outputs["test_labels"],
                current_round=current_round,
                epoch=None,
                run_id=None,
                encoded_type="testEncoded")

            accuracy = test_outputs["accuracy"]

            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(
                accuracy=accuracy,
                current_round=kwargs['current_round'],
                epoch=0,
                run_id=None)

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