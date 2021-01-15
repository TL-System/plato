"""
The training and testing loop.
"""

import logging
import os
import torch

import numpy as np

from models.base import Model
from config import Config
from trainers import base, optimizers


class Trainer(base.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""
    def __init__(self, model: Model):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train. Must be a models.base.Model subclass.
        """

        # Use distributed data parallelism if multiple GPUs are available.
        if Config().is_distributed():
            logging.info("Turning on Distributed Data Parallelism...")

            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = Config().DDP_port()
            torch.distributed.init_process_group(
                'nccl', rank=0, world_size=Config().world_size())

            # DistributedDataParallel divides and allocate a batch of data to all
            # available GPUs since device_ids are not set
            self.model = torch.nn.parallel.DistributedDataParallel(
                module=model)
        else:
            self.model = model

            self.device = Config().device()
            self.model.to(self.device)

            self.model.train()

    def save_model(self):
        """Saving the model to a file."""
        model_type = Config().trainer.model
        model_dir = './models/pretrained/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{model_dir}{model_type}.pth'
        torch.save(self.model.state_dict(), model_path)
        logging.info('Model saved to %s.', model_path)

    def load_model(self, model_type):
        """Loading pre-trained model weights from a file."""
        model_path = f'./models/pretrained/{model_type}.pth'
        self.model.load_state_dict(torch.load(model_path))

    def extract_weights(self):
        """Extract weights from the model."""
        weights = []
        for name, weight in self.model.to(
                torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                weights.append((name, weight.data))

        return weights

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = []
            for i, (name, current_weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = current_weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        updated_state_dict = {}
        for name, weight in weights:
            updated_state_dict[name] = weight

        self.model.load_state_dict(updated_state_dict, strict=False)

    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

        log_interval = 10
        batch_size = Config().trainer.batch_size
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = Config().trainer.epochs

        # Initializing the optimizer
        optimizer = optimizers.get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if Config().trainer.lr_gamma == 0.0 or Config(
        ).trainer.lr_milestone_steps == '':
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch)
        else:
            lr_schedule = None

        for epoch in range(1, epochs + 1):
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                optimizer.zero_grad()
                if cut_layer is None:
                    loss = self.model.loss_criterion(self.model(examples),
                                                     labels)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)
                    loss = self.model.loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if lr_schedule is not None:
                    lr_schedule.step()

                if batch_id % log_interval == 0:
                    logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, epochs, loss.item()))

    def test(self, testset, batch_size, cut_layer=None):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        batch_size: the batch size used for testing.
        cut_layer (optional): The layer which testing should start from.
        """

        self.model.to(self.device)
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

        return accuracy
