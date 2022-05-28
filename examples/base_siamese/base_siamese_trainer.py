"""
Implement the trainer for base siamese method.

"""
import os
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from opacus.privacy_engine import PrivacyEngine

from sklearn.neural_network import MLPClassifier

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
        output1, output2 = outputs
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) + (label) *
            torch.pow(torch.clamp(self.margin -
                                  euclidean_distance, min=0.0), 2))

        return loss_contrastive


#   Here are two stages to test the contrastive learning:
#   - The first stage is that when training the contrastive learning methods,
#      the model is tested based on the contrastive evaluation, i.e. similarity.
#   - The second stage is to test the learned representation contrastive methods.
#      the learned representation is utilized for downstream tasks, such as classification
#      thus, the task-specific metric, such as the accuracy should be used.
#
#      For example,
#     In the second stage, there are two methods to test the contrastive learning,
#       1- linear evaluation,
#       first learn representations from the framework.
#       Then, train a new linear classifier on the frozen representations.


class Trainer(basic.Trainer):

    def __init__(self, model=None):
        super().__init__(model)

    def loss_criterion(self, model):
        """ The loss computation. """
        # define the loss computation instance
        defined_margin = Config().trainer.margin
        constrative_loss_computer = ContrastiveLoss(margin=defined_margin)
        return constrative_loss_computer

    def train_loop(self, config, trainset, sampler, cut_layer):
        """ The default training loop when a custom training loop is not supplied. """
        batch_size = config['batch_size']
        log_interval = 10
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

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if hasattr(config, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        for epoch in range(1, epochs + 1):
            # Use a default training loop
            for batch_id, (examples1, examples2,
                           labels) in enumerate(train_loader):
                examples1, examples2, labels = examples1.to(
                    self.device), examples2.to(self.device), labels.to(
                        self.device)
                examples = (examples1, examples2)

                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)

                if 'create_graph' in config:
                    loss.backward(create_graph=config['create_graph'])
                else:
                    loss.backward()

                optimizer.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())

            if lr_schedule is not None:
                lr_schedule.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if hasattr(Config().server,
                       'request_update') and Config().server.request_update:
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

    def test_process(self, config, testset, sampler=None):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
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
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        sampler=sampler.get())

                samples_feature = []
                samples_label = []
                with torch.no_grad():
                    for examples, labels in test_loader:
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        outputs = self.model.forward_once(examples)
                        samples_feature.extend(
                            outputs.data.cpu().numpy().tolist())
                        samples_label.extend(
                            labels.data.cpu().numpy().tolist())

                    # define a simple MLP classifier
                    clf = MLPClassifier(solver='adam',
                                        alpha=1e-5,
                                        hidden_layer_sizes=(10, 10),
                                        random_state=1)
                    clf.fit(samples_feature, samples_label)
                    accuracy = clf.score(samples_feature, samples_label)

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy
