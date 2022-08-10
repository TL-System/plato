"""
Implementation of our contrastive adaptation trainer.

"""

import os
import logging

import numpy as np
from tqdm import tqdm
import torch

from plato.config import Config
from plato.trainers import contrastive_ssl
from plato.utils import optimizers

from plato.utils.checkpoint_operator import perform_client_checkpoint_saving
from plato.utils.checkpoint_operator import get_client_checkpoint_operator
from plato.utils.arrange_saving_name import get_format_name
from plato.utils.checkpoint_operator import perform_client_checkpoint_loading

from pFLCMA_losses import pFLCMALoss


class Trainer(contrastive_ssl.Trainer):
    """ The federated learning trainer for the BYOL client. """

    def freeze_model(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False

    def active_model(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = True

    @staticmethod
    def loss_criterion(model, config):
        """ The loss computation. """
        temperature = config['temperature']
        base_temperature = config['base_temperature']
        contrast_mode = config['contrast_mode']

        similarity_lambda = config[
            'similarity_lambda'] if 'similarity_lambda' in config else 0.0

        ntx_lambda = config['ntx_lambda'] if 'ntx_lambda' in config else 0.0

        prototype_contrastive_repr_lambda = config[
            'prototype_contrastive_repr_lambda'] if 'prototype_contrastive_repr_lambda' in config else 0.0

        meta_lambda = config['meta_lambda'] if 'meta_lambda' in config else 0.0

        meta_contrastive_lambda = config[
            'meta_contrastive_lambda'] if 'meta_contrastive_lambda' in config else 0.0

        to_compute_losses = config['cma_losses']

        to_compute_losses = to_compute_losses.split(",")

        contrastive_adaptation_criterion = pFLCMALoss(
            losses=to_compute_losses,
            similarity_lambda=similarity_lambda,
            ntx_lambda=ntx_lambda,
            prototype_contrastive_repr_lambda=prototype_contrastive_repr_lambda,
            meta_lambda=meta_lambda,
            meta_contrastive_lambda=meta_contrastive_lambda,
            perform_label_distortion=False,
            label_distrotion_type="random",
            temperature=temperature,
            base_temperature=base_temperature,
            contrast_mode=contrast_mode)

        return contrastive_adaptation_criterion

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
        config['current_round'] = current_round
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
                optimizer_state_dict=eval_optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=0,
                base_epoch=0,
                prefix="personalized")

            accuracy, _, _ = self.perform_evaluation_op(test_loader)
            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(accuracy=accuracy,
                                                  current_round=current_round,
                                                  epoch=0,
                                                  run_id=None)

            # Initializing the loss criterion
            _eval_loss_criterion = getattr(self, "pers_loss_criterion", None)
            if callable(_eval_loss_criterion):
                eval_loss_criterion = self.pers_loss_criterion(
                    self.personalized_model)
            else:
                eval_loss_criterion = torch.nn.CrossEntropyLoss()

            self.personalized_model.to(self.device)
            self.model.encoder.to(self.device)

            self.freeze_model(self.model.encoder)
            self.active_model(self.personalized_model)

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
                self.model.encoder.train()
                local_progress = tqdm(eval_train_loader,
                                      desc=f'Epoch {epoch}/{pers_epochs+1}',
                                      disable=True)

                for _, (examples, labels) in enumerate(local_progress):
                    examples, labels = examples.to(self.device), labels.to(
                        self.device)
                    # Clear the previous gradient
                    eval_optimizer.zero_grad()

                    # Extract representation from the trained
                    # encoder of ssl.
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

                    accuracy, _, _ = self.perform_evaluation_op(test_loader)
                    # save the personaliation accuracy to the results dir
                    self.checkpoint_personalized_accuracy(
                        accuracy=accuracy,
                        current_round=current_round,
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
                        optimizer_state_dict=eval_optimizer.state_dict(),
                        lr_schedule_state_dict=lr_schedule.state_dict(),
                        present_epoch=epoch,
                        base_epoch=epoch,
                        prefix="personalized")

                lr_schedule.step()

            accuracy, test_encoded, test_labels = self.perform_evaluation_op(
                test_loader)

        except Exception as testing_exception:
            logging.info("Evaluation Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        # save the personalized model for current round
        # to the model dir of this client
        if 'max_concurrency' in config:

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
