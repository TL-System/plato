"""
A personalized federated learning trainer using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import os
import logging
import warnings

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers

from plato.utils.checkpoint_operator import perform_client_checkpoint_saving


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def freeze_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = False

    def active_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = True

    def train_one_epoch(self, config, epoch, defined_model, optimizer,
                        loss_criterion, train_data_loader, epoch_loss_meter,
                        batch_loss_meter):
        defined_model.train()
        epochs = config['epochs']

        # load the local update epochs for head optimization
        head_epochs = config[
            'head_epochs'] if 'head_epochs' in config else epochs - 1

        iterations_per_epoch = len(train_data_loader)
        # default not to perform any logging
        epoch_log_interval = epochs + 1
        batch_log_interval = iterations_per_epoch

        if "epoch_log_interval" in config:
            epoch_log_interval = config['epoch_log_interval']
        if "batch_log_interval" in config:
            batch_log_interval = config['batch_log_interval']

        # As presented in Section 3 of the FedRep paper, the head is optimized
        # for (epochs - 1) while freezing the representation.
        if epoch <= head_epochs:
            self.freeze_model(defined_model, param_prefix="encoder")
            self.active_model(defined_model, param_prefix="clf_fc")

        # The representation will then be optimized for only one epoch
        if epoch > head_epochs:
            self.freeze_model(defined_model, param_prefix="clf_fc")
            self.active_model(defined_model, param_prefix="encoder")

        # print("epoch: ", epoch)
        # for name, param in defined_model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

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

    def perform_test_op(self, test_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
        defined_model.eval()
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

    def pers_train_one_epoch(
        self,
        config,
        kwargs,
        epoch,
        defined_model,
        pers_optimizer,
        lr_schedule,
        pers_loss_criterion,
        pers_train_loader,
        test_loader,
        epoch_loss_meter,
    ):
        """ Performing one epoch of learning for the personalization. """
        personalized_model_name = Config().trainer.personalized_model_name
        current_round = kwargs['current_round']

        epoch_loss_meter.reset()
        defined_model.train()

        pers_epochs = config["pers_epochs"]
        epoch_log_interval = pers_epochs + 1
        epoch_model_log_interval = pers_epochs + 1

        if "pers_epoch_log_interval" in config:
            epoch_log_interval = config['pers_epoch_log_interval']

        if "pers_epoch_model_log_interval" in config:
            epoch_model_log_interval = config['pers_epoch_model_log_interval']

        local_progress = tqdm(pers_train_loader,
                              desc=f'Epoch {epoch}/{pers_epochs+1}',
                              disable=True)
        # encoded data
        train_encoded = list()
        train_labels = list()

        self.freeze_model(defined_model, param_prefix="encoder")
        self.active_model(defined_model, param_prefix="clf_fc")

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            # preds = self.personalized_model(examples)
            features = defined_model.encoder(examples)
            preds = defined_model.clf_fc(features)

            loss = pers_loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss.data.item(), labels.size(0))

            # save the encoded train data of current epoch
            if epoch == pers_epochs:
                train_encoded.append(features)
                train_labels.append(labels)

            local_progress.set_postfix({
                'lr': lr_schedule,
                "loss": epoch_loss_meter.val,
                'loss_avg': epoch_loss_meter.avg
            })

        if (epoch - 1) % epoch_log_interval == 0 or epoch == pers_epochs:
            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id, epoch, pers_epochs, epoch_loss_meter.avg)

            test_outputs = self.perform_test_op(test_loader, defined_model)

            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(
                accuracy=test_outputs["accuracy"],
                current_round=current_round,
                epoch=epoch,
                run_id=None)

        if (epoch - 1) % epoch_model_log_interval == 0 or epoch == pers_epochs:
            # the model generated during each round will be stored in the
            # checkpoints
            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=personalized_model_name,
                model_state_dict=defined_model.state_dict(),
                config=config,
                kwargs=kwargs,
                optimizer_state_dict=pers_optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=epoch,
                base_epoch=epoch,
                prefix="personalized")

        if epoch == pers_epochs:
            self.checkpoint_encoded_samples(encoded_samples=train_encoded,
                                            encoded_labels=train_labels,
                                            current_round=current_round,
                                            epoch=epoch,
                                            run_id=None,
                                            encoded_type="trainEncoded")
            self.checkpoint_encoded_samples(
                encoded_samples=test_outputs["test_encoded"],
                encoded_labels=test_outputs["test_labels"],
                current_round=current_round,
                epoch=epoch,
                run_id=None,
                encoded_type="testEncoded")
