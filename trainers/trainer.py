"""
The training and testing loops for PyTorch.
"""

import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

from models.base import Model
from config import Config
from trainers import base, optimizers


class Trainer(base.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""
    def __init__(self, model: Model, client_id=0):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train. Must be a models.base.Model subclass.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(client_id)

        # Use data parallelism if multiple GPUs are available and the configuration specifies it
        if Config().is_parallel():
            logging.info("Using Data Parallelism.")
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model)
        else:
            self.model = model

    def zeros(self, shape):
        """Returns a MindSpore zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None):
        """Saving the model to a file."""
        model_type = Config().trainer.model
        model_dir = Config().model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_type}.pth'

        torch.save(self.model.state_dict(), model_path)

        if self.client_id == 0:
            logging.info("[Server #%s] Model saved to %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%s] Model saved to %s.", self.client_id,
                         model_path)

    def load_model(self, filename=None):
        """Loading pre-trained model weights from a file."""
        model_type = Config().trainer.model
        model_dir = Config().model_dir

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_type}.pth'

        if self.client_id == 0:
            logging.info("[Server #%s] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%s] Loading a model from %s.",
                         self.client_id, model_path)

        self.model.load_state_dict(torch.load(model_path))

    @staticmethod
    def save_accuracy(accuracy, filename):
        """Saving the test accuracy to a file."""
        model_dir = Config().model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        accuracy_path = f"{model_dir}{filename}"

        with open(accuracy_path, 'w') as file:
            file.write(str(accuracy))

    @staticmethod
    def load_accuracy(filename):
        """Loading the test accuracy from a file."""
        model_dir = Config().model_dir
        accuracy_path = f"{model_dir}{filename}"

        with open(accuracy_path, 'r') as file:
            accuracy = float(file.read())

        return accuracy

    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.state_dict()

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                delta = current_weight - baseline
                update[name] = delta
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)

    @staticmethod
    def train_process(rank, self, config, trainset, cut_layer=None):  # pylint: disable=unused-argument
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        log_interval = 10
        batch_size = config['batch_size']
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        loss_criterion = nn.CrossEntropyLoss()

        # Initializing the optimizer
        optimizer = optimizers.get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if 'lr_gamma' in config or 'lr_milestone_steps' in config:
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        for epoch in range(1, epochs + 1):
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_schedule.step()

                if config['optimizer'] == 'FedProx':
                    optimizer.params_state_update()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                            format(os.getpid(), epoch, epochs, batch_id,
                                   len(train_loader), loss.data.item()))
                    else:
                        logging.info(
                            "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                            format(self.client_id, epoch, epochs, batch_id,
                                   len(train_loader), loss.data.item()))
        self.model.cpu()

        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{config['experiment_id']}.pth"
        self.save_model(filename)

    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        rank: Required by torch.multiprocessing to spawn processes. Unused.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        self.start_training()

        config = Config().trainer._asdict()
        config['experiment_id'] = Config().experiment_id

        mp.spawn(Trainer.train_process,
                 args=(
                     self,
                     config,
                     trainset,
                     cut_layer,
                 ),
                 join=True)

        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{Config().experiment_id}.pth"
        self.load_model(filename)
        self.pause_training()

    @staticmethod
    def test_process(rank, self, config, testset):  # pylint: disable=unused-argument
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        rank: Required by torch.multiprocessing to spawn processes. Unused.
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
        self.model.to(self.device)
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.cpu()

        accuracy = correct / total
        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{config['experiment_id']}.acc"
        Trainer.save_accuracy(accuracy, filename)

    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
        self.start_training()
        config = Config().trainer._asdict()
        config['experiment_id'] = Config().experiment_id

        mp.spawn(Trainer.test_process,
                 args=(
                     self,
                     config,
                     testset,
                 ),
                 join=True)

        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{Config().experiment_id}.acc"
        accuracy = Trainer.load_accuracy(filename)

        self.pause_training()
        return accuracy
