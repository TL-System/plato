"""
A personalized federated learning trainer for the FedPer method.

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

        return train_encoded, train_labels, test_outputs[
            "test_encoded"], test_outputs["test_labels"]

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

        # also record the encoded data in the first epoch
        if epoch == 1:
            train_encoded, train_labels, test_encoded, test_labels = self.obtain_encoded_data(
                defined_model, pers_train_loader, test_loader)
            self.checkpoint_encoded_samples(encoded_samples=train_encoded,
                                            encoded_labels=train_labels,
                                            current_round=current_round,
                                            epoch=epoch - 1,
                                            run_id=None,
                                            encoded_type="trainEncoded")
            self.checkpoint_encoded_samples(encoded_samples=test_encoded,
                                            encoded_labels=test_labels,
                                            current_round=current_round,
                                            epoch=epoch - 1,
                                            run_id=None,
                                            encoded_type="testEncoded")

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
