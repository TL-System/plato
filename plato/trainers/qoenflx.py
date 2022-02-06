"""
A customized trainer for QoENFLX
"""
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.datasources import qoenflx
from plato.trainers import basic
from plato.utils import optimizers
from scipy.stats import spearmanr


class Trainer(basic.Trainer):
    """A custom trainer for QoENFLX. """
    def __init__(self, model=None):
        super().__init__(model)

    def test_model(self, config, testset):
        test_loader = qoenflx.DataSource.get_test_loader(
            config['batch_size'], testset)

        with torch.no_grad():
            for examples in test_loader:
                x1 = torch.autograd.Variable(examples['VQA'].to(self.device))
                x2 = torch.autograd.Variable(examples['R1'].to(self.device))
                x3 = torch.autograd.Variable(examples['R2'].to(self.device))
                x4 = torch.autograd.Variable(examples['Mem'].to(self.device))
                x5 = torch.autograd.Variable(examples['Impair'].to(
                    self.device))
                labels = torch.autograd.Variable(examples['label'].to(
                    self.device)).float()

                outputs = self.model(x1, x2, x3, x4, x5)

                srocc = spearmanr(labels.cpu(), outputs.cpu())[0]

        return srocc

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        log_interval = 10
        batch_size = config['batch_size']

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        train_loader = qoenflx.DataSource.get_train_loader(
            trainset=trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

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
            for batch_id, examples in enumerate(train_loader):
                x1 = torch.autograd.Variable(examples['VQA'].to(self.device))
                x2 = torch.autograd.Variable(examples['R1'].to(self.device))
                x3 = torch.autograd.Variable(examples['R2'].to(self.device))
                x4 = torch.autograd.Variable(examples['Mem'].to(self.device))
                x5 = torch.autograd.Variable(examples['Impair'].to(
                    self.device))
                labels = torch.autograd.Variable(examples['label'].to(
                    self.device)).float()

                optimizer.zero_grad()

                outputs = self.model(x1, x2, x3, x4, x5)

                loss = F.mse_loss(outputs, labels)

                loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_schedule.step()

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

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

        # Save the training loss of the last epoch in this round
        model_name = config['model_name']
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.loss"
        Trainer.save_loss(loss.data.item(), filename)

    @staticmethod
    def save_loss(loss, filename=None):
        """Saving the training loss to a file."""
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            loss_path = f"{model_dir}{filename}"
        else:
            loss_path = f'{model_dir}{model_name}.loss'

        with open(loss_path, 'w') as file:
            file.write(str(loss))

    @staticmethod
    def load_loss(filename=None):
        """Loading the training loss from a file."""
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if filename is not None:
            loss_path = f"{model_dir}{filename}"
        else:
            loss_path = f'{model_dir}{model_name}.loss'

        with open(loss_path, 'r') as file:
            loss = float(file.read())

        return loss
