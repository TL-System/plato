"""
The training and testing loops for PyTorch.
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer for personalized FL using MAML algorithm."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)
        self.test_personalization = False

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """A custom training loop for personalized FL."""
        batch_size = config["batch_size"]
        log_interval = 10

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
            )

        epochs = config["epochs"]

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer for the second stage of MAML
        # The learning rate here is the meta learning rate (beta)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=Config().trainer.meta_learning_rate,
            momentum=Config().trainer.momentum,
            weight_decay=Config().trainer.weight_decay,
        )

        # Initializing the learning rate schedule, if necessary
        # Initializing the schedule for meta learning rate, if necessary
        if "meta_lr_schedule" in config:
            meta_lr_schedule = optimizers.get_lr_schedule(
                optimizer, len(train_loader), train_loader
            )
        else:
            meta_lr_schedule = None

        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, epochs + 1):
            # Copy the current model due to using MAML
            current_model = copy.deepcopy(self.model)
            # Sending this model to the device used for training
            current_model.to(self.device)
            current_model.train()

            # Initializing the optimizer for the first stage of MAML
            # The learning rate here is the alpha in the paper
            local_optimizer = torch.optim.SGD(
                current_model.parameters(),
                lr=Config().parameters.optimizer.lr,
                momentum=Config().trainer.momentum,
                weight_decay=Config().trainer.weight_decay,
            )

            # Initializing the learning rate schedule, if necessary
            if "lr_schedule" in config:
                lr_schedule = optimizers.get_lr_schedule(
                    local_optimizer, len(train_loader), train_loader
                )
            else:
                lr_schedule = None

            # The first stage of MAML
            # Use half of the training dataset
            self.training_per_stage(
                1,
                local_optimizer,
                lr_schedule,
                train_loader,
                current_model,
                loss_criterion,
                log_interval,
                epoch,
                epochs,
            )

            # The second stage of MAML
            # Use the other half of the training dataset
            self.training_per_stage(
                2,
                optimizer,
                meta_lr_schedule,
                train_loader,
                self.model,
                loss_criterion,
                log_interval,
                epoch,
                epochs,
            )

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

    def training_per_stage(
        self,
        stage_id,
        optimizer,
        lr_schedule,
        train_loader,
        cut_layer,
        training_model,
        loss_criterion,
        log_interval,
        epoch,
        epochs,
    ):
        """The training process of the two stages of MAML."""
        if stage_id == 1:
            batch_id_range = [i for i in range(int(len(train_loader) / 2))]
        elif stage_id == 2:
            batch_id_range = [
                i for i in range(int(len(train_loader) / 2), len(train_loader))
            ]

        for batch_id, (examples, labels) in enumerate(train_loader):
            if batch_id in batch_id_range:
                examples, labels = examples.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = training_model(examples)
                loss = loss_criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_scheduler.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(),
                            epoch,
                            epochs,
                            batch_id,
                            len(train_loader),
                            loss.data.item(),
                        )
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id,
                            epoch,
                            epochs,
                            batch_id,
                            len(train_loader),
                            loss.data.item(),
                        )

            if lr_schedule is not None:
                lr_scheduler.step()

    def test_process(self, config, testset, sampler=None, **kwargs):
        """A customized testing loop for personalized FL."""
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            correct = 0
            total = 0
            if sampler is None:
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=config["batch_size"], shuffle=False
                )
            # Use a testing set following the same distribution as the training set
            else:
                test_loader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    sampler=sampler.get(),
                )

            # Test its personalized model during personalization test
            if self.test_personalization:
                logging.info("[Client #%d] Personalizing its model.", self.client_id)
                # Generate a training set for personalization
                # by randomly choose one batch from test set
                random_batch_id = np.random.randint(0, len(test_loader))
                for batch_id, (examples, labels) in enumerate(test_loader):
                    if batch_id == random_batch_id:
                        personalize_train_set = [examples, labels]

                personalized_model = self.personalize_client_model(
                    personalize_train_set
                )

                personalized_model.eval()

                with torch.no_grad():
                    for batch_id, (examples, labels) in enumerate(test_loader):
                        # Aviod using the batch used to generate the personalized model when testing
                        if batch_id != random_batch_id:
                            examples, labels = (
                                examples.to(self.device),
                                labels.to(self.device),
                            )

                            outputs = personalized_model(examples)

                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

            # Test its trained local model during training of the global model
            else:
                with torch.no_grad():
                    for examples, labels in test_loader:
                        examples, labels = (
                            examples.to(self.device),
                            labels.to(self.device),
                        )

                        outputs = self.model(examples)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

            accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        if self.test_personalization:
            logging.info("[Client #%d] Finished personalization test.", self.client_id)
        else:
            self.model.cpu()

        if "max_concurrency" in config:
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def personalize_client_model(self, personalize_train_set):
        """ "Run one step of gradient descent to personalze a client's model."""
        personalized_model = copy.deepcopy(self.model)
        personalized_model.to(self.device)
        personalized_model.train()

        loss_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            personalized_model.parameters(),
            lr=Config().parameters.optimizer.lr,
            momentum=Config().trainer.momentum,
            weight_decay=Config().trainer.weight_decay,
        )

        examples = personalize_train_set[0]
        labels = personalize_train_set[1]

        examples, labels = examples.to(self.device), labels.to(self.device)
        optimizer.zero_grad()

        outputs = personalized_model(examples)

        loss = loss_criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        return personalized_model
