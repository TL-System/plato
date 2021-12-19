"""
The training and testing loops for PyTorch.
"""
import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A federated learning trainer for personalized FL using MAML algorithm."""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)
        self.test_personalization = False

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload."""

        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                _train_loader = getattr(self, "train_loader", None)

                if callable(_train_loader):
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=sampler.get())

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                # Initializing the optimizer for the second stage of MAML
                # The learning rate here is the meta learning rate (beta)
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=Config().trainer.meta_learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config().trainer.weight_decay)

                # Initializing the schedule for meta learning rate, if necessary
                if hasattr(config, 'meta_lr_schedule'):
                    meta_lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    meta_lr_schedule = None

                for epoch in range(1, epochs + 1):
                    # Copy the current model due to using MAML
                    current_model = copy.deepcopy(self.model)
                    # Sending this model to the device used for training
                    current_model.to(self.device)
                    current_model.train()

                    # Initializing the optimizer for the first stage of MAML
                    # The learning rate here is the alpha in the paper
                    temp_optimizer = torch.optim.SGD(
                        current_model.parameters(),
                        lr=Config().trainer.learning_rate,
                        momentum=Config().trainer.momentum,
                        weight_decay=Config().trainer.weight_decay)

                    # Initializing the learning rate schedule, if necessary
                    if hasattr(config, 'lr_schedule'):
                        lr_schedule = optimizers.get_lr_schedule(
                            temp_optimizer, iterations_per_epoch, train_loader)
                    else:
                        lr_schedule = None

                    # The first stage of MAML
                    # Use half of the training dataset
                    self.training_per_stage(1, temp_optimizer, lr_schedule,
                                            train_loader, cut_layer,
                                            current_model, loss_criterion,
                                            log_interval, config, epoch,
                                            epochs)

                    # The second stage of MAML
                    # Use the other half of the training dataset
                    self.training_per_stage(2, optimizer, meta_lr_schedule,
                                            train_loader, cut_layer,
                                            self.model, loss_criterion,
                                            log_interval, config, epoch,
                                            epochs)

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()

    def training_per_stage(self, stage_id, optimizer, lr_schedule,
                           train_loader, cut_layer, training_model,
                           loss_criterion, log_interval, config, epoch,
                           epochs):
        """The training process of the two stages of MAML."""
        if stage_id == 1:
            batch_id_range = [i for i in range(int(len(train_loader) / 2))]
        elif stage_id == 2:
            batch_id_range = [
                i for i in range(int(len(train_loader) / 2), len(train_loader))
            ]

        for batch_id, (examples, labels) in enumerate(train_loader):
            if batch_id in batch_id_range:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = training_model(examples)
                else:
                    outputs = training_model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_schedule.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss of stage {}: {:.6f}"
                            .format(os.getpid(), epoch, epochs, batch_id,
                                    len(train_loader), stage_id,
                                    loss.data.item()))
                    else:
                        if hasattr(config, 'use_wandb'):
                            wandb.log({"batch loss": loss.data.item()})

                        logging.info(
                            "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss of stage {}: {:.6f}"
                            .format(self.client_id, epoch, epochs, batch_id,
                                    len(train_loader), stage_id,
                                    loss.data.item()))

    def test_process(self, config, testset):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        """
        if not self.test_personalization:
            self.model.to(self.device)
            self.model.eval()

        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=config['batch_size'], shuffle=False)

                correct = 0
                total = 0

                # Test its personalized model during personalization test
                if self.test_personalization:
                    logging.info("[Client #%d] Personalizing its model.",
                                 self.client_id)
                    # Generate a training set for personalization
                    # by randomly choose one batch from test set
                    random_batch_id = random.randint(0, len(test_loader) - 1)
                    for batch_id, (examples, labels) in enumerate(test_loader):
                        if batch_id == random_batch_id:
                            personalize_train_set = [examples, labels]

                    personalized_model = self.personalize_client_model(
                        personalize_train_set)

                    personalized_model.eval()

                    with torch.no_grad():
                        for batch_id, (examples,
                                       labels) in enumerate(test_loader):
                            # Aviod using the batch used to generate the personalized model when testing
                            if batch_id != random_batch_id:
                                examples, labels = examples.to(
                                    self.device), labels.to(self.device)

                                outputs = personalized_model(examples)

                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                # Test its trained local model during training of the global model
                else:
                    with torch.no_grad():
                        for examples, labels in test_loader:
                            examples, labels = examples.to(
                                self.device), labels.to(self.device)

                            outputs = self.model(examples)

                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        if not self.test_personalization:
            self.model.cpu()
        else:
            logging.info("[Client #%d] Finished personalization test.",
                         self.client_id)

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def personalize_client_model(self, personalize_train_set):
        """"Run one step of gradient descent to personalze a client's model. """
        personalized_model = copy.deepcopy(self.model)
        personalized_model.to(self.device)
        personalized_model.train()

        loss_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(personalized_model.parameters(),
                                    lr=Config().trainer.learning_rate,
                                    momentum=Config().trainer.momentum,
                                    weight_decay=Config().trainer.weight_decay)

        examples = personalize_train_set[0]
        labels = personalize_train_set[1]

        examples, labels = examples.to(self.device), labels.to(self.device)
        optimizer.zero_grad()

        outputs = personalized_model(examples)

        loss = loss_criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        return personalized_model
