"""
A personalized federated learning trainer using FedRep.

"""

import os
import logging
import warnings

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
from plato.trainers import pers_basic
from plato.utils import optimizers


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

        self.freeze_model(defined_model, param_prefix="encoder")
        self.active_model(defined_model, param_prefix="clf_fc")

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
        eval_outputs, _ = super().on_start_pers_train(defined_model,
                                                      model_name, data_loader,
                                                      epoch, global_epoch,
                                                      config, optimizer,
                                                      lr_schedule)
        self.checkpoint_encoded_samples(
            encoded_samples=eval_outputs['encoded_samples'],
            encoded_labels=eval_outputs['loaded_labels'],
            current_round=current_round,
            epoch=epoch,
            run_id=None,
            encoded_type="testEncoded")

        return eval_outputs, _

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
        current_round = config['current_round']
        eval_outputs = super().on_end_pers_train_epoch(
            defined_model, model_name, data_loader, epoch, global_epoch,
            config, optimizer, lr_schedule, epoch_loss_meter)
        if eval_outputs:
            self.checkpoint_encoded_samples(
                encoded_samples=eval_outputs['encoded_samples'],
                encoded_labels=eval_outputs['loaded_labels'],
                current_round=current_round,
                epoch=epoch,
                run_id=None,
                encoded_type="testEncoded")

        return eval_outputs
