"""
A federated semi-supervised learning trainer using FedMatch.
Reference:
Jeong et al., "Federated Semi-supervised learning with inter-client consistency & disjoint learning", in the Proceedings of ICLR 2021.
https://arxiv.org/pdf/2006.12097.pdf 
"""

import torch
from torch import optim
from plato.config import Config
from plato.trainers import basic
import asyncio
import logging
import multiprocessing as mp
import os
import time
from typing import Tuple
import random
import numpy as np
import torch.nn as nn
import wandb
from plato.models import registry as models_registry
from plato.utils import optimizers
from scipy.ndimage.interpolation import shift
from PIL import Image
from plato.config import Config


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client. """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.
        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        # variables to be added when used
        super().__init__(model)
        self.confident = 0
        self.lambda_s = 10
        self.lambda_i = 1e-2
        self.lambda_l1 = 1e-4
        self.lambda_l2 = 10
        #self.kl_divergence = nn.functional.kl_div()

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.
        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        if 'use_wandb' in config:
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
                """obtain labeled and unlabeled dataset"""
                #print('===========len of train loader is : ',
                #len(train_loader))
                #print('type of train_loader is : ', type(train_loader))
                train_loader_s, train_loader_u = torch.utils.data.random_split(
                    train_loader, [125, 500])  # rewrite with ratio
                print("==========================Type of train_loader: ",
                      type(train_loader))
                print("Type of train_loader_s: ", type(train_loader_s))

                # iterations_per_epoch = np.ceil(len(trainset) /
                #                               batch_size).astype(int) # variable iterations_per_epoch is computed for lr_scheduler;
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()
                """"""
                # Initializing the loss criterion for supervised learning
                _loss_criterion_s = getattr(self, "loss_criterion_s", None)
                if callable(_loss_criterion_s):
                    loss_criterion_s = self.loss_criterion_s(self.model)
                else:
                    loss_criterion_s = nn.CrossEntropyLoss()

                # Initializing the loss criterion for unsupervised learning
                _loss_criterion_u = getattr(self, "loss_criterion_u", None)
                if callable(_loss_criterion_u):
                    loss_criterion_u = self.loss_criterion_u(self.model)
                else:
                    loss_criterion_u = nn.CrossEntropyLoss()

                # Initializing the optimizer
                #get_optimizer = getattr(self, "get_optimizer",
                #                        optimizers.get_optimizer)
                """
                print("Params of disjoint model: ", self.model.parameters())
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)
                """

                optimizer_s = optim.SGD(
                    [list(self.model.parameters())[0]],
                    lr=Config().trainer.learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config(
                    ).trainer.weight_decay)  #get_optimizer(self.model.psis)
                optimizer_u = optim.SGD(
                    [list(self.model.parameters())[1]],
                    lr=Config().trainer.learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config(
                    ).trainer.weight_decay)  #get_optimizer(self.model.sigmas)
                """
                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None
                """
                #self.model.acho()

                for epoch in range(1, epochs + 1):
                    self.confident = 0
                    for batch_id, (examples, labels) in enumerate(
                            train_loader):  # batch_id is used for logging
                        #(examples_unlabeled, discards) in zip(train_loader_s, train_loader_u):
                        #######################
                        # supervised learning
                        #######################
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        optimizer_s.zero_grad()

                        if cut_layer is None:
                            outputs_s = self.model(examples)
                        else:
                            outputs_s = self.model.forward_from(
                                examples, cut_layer)

                        loss_s = loss_criterion_s(outputs_s,
                                                  labels) * self.lambda_s

                        loss_s.backward()

                        optimizer_s.step()

                        #######################
                        # unsupervised learning
                        #######################
                    for batch_id, (examples_unlabeled, labels) in enumerate(
                            train_loader_u):  #why not accessiable
                        #pseduo_labels = self.model(self.loader.scale(examples_unlabeled))
                        optimizer_u.zero_grad()
                        """
                        if cut_layer is None:
                            outputs_u = self.model(examples)
                        else:
                            outputs_u = self.model.forward_from(
                                examples, cut_layer)
                        """
                        loss_u, _confident = self.loss_unsupervised(
                            examples_unlabeled, loss_criterion_u)

                        loss_u.backward()

                        optimizer_u.step()

                        self.confident += _confident

                        #if lr_schedule is not None:
                        #    lr_schedule.step()
                        """
                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss_s.data.item()+loss_u.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss_s.data.item()+loss_u.data.item()})
                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss_s.data.item()+loss_u.data.item()))
                        """

                    #if hasattr(optimizer, "params_state_update"):
                    #   optimizer.params_state_update()

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        self.model.cpu()

        model_type = config['model_name']
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
        self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()

    def loss_unsupervised(self,
                          unlabled_samples,
                          loss_criterion_u,
                          cut_layer=None):

        loss_u = 0
        """
        # Initializing the loss criterion for unsupervised learning
        _loss_criterion_u = getattr(self, "loss_criterion_u", None)
        if callable(_loss_criterion_u):
            loss_criterion_u = self.loss_criterion_u(self.model)
        else:
            loss_criterion_u = nn.CrossEntropyLoss()
        """
        # Make predictions with local model
        if cut_layer is None:
            y_pred = self.model(self.scale(unlabled_samples))
        else:
            y_pred = self.model.forward_from(self.scale(unlabled_samples),
                                             cut_layer)

        _confident = np.where(
            np.max(y_pred.numpy(), axis=1) >= self.confident)[0]

        if len(_confident) > 0:
            # Inter-client consistency
            samples_confident = self.scale(unlabled_samples[_confident])
            y_pred = torch.gather(y_pred, 1, _confident)

            y_preds = [
                rm(samples_confident).numpy()
                for rid, rm in enumerate(self.helpers)
            ]  # where find helpers

            for _, pred in enumerate(y_preds):
                loss_u += (nn.functional.kl_div(pred, y_pred) /
                           len(y_preds)) * self.lambda_i

            # Agreement-based Pseudo Labeling
            if cut_layer is None:
                y_hard = self.model(
                    self.scale(
                        self.augment(unlabled_samples[_confident],
                                     soft=False)))
            else:
                y_hard = self.model.forward_from(
                    self.scale(
                        self.augment(unlabled_samples[_confident],
                                     soft=False)), cut_layer)

            y_pseu = self.agreement_based_labeling(y_pred, y_preds)
            loss_u += loss_criterion_u(y_pseu, y_hard) * self.lambda_a

        # Regularization
        for lid, psi in enumerate(self.psi):  # psi & sig?
            # l1 regularization
            loss_u += torch.reduce_sum(torch.abs(psi)) * self.lambda_l1
            # l2 regularization
            loss_u += torch.reduce_sum(
                torch.math.square(self.sig[lid] - psi)) * self.lambda_l2

        return loss_u, len(
            _confident
        )  # confident is only used for logging in orignal code so we can get rid of it here.

    def scale(self, x):
        x = x.astype(np.float32) / 255

    def augment(self, images, soft=True):
        if soft:
            indices = np.arange(len(images)).tolist()
            sampled = random.sample(indices, int(round(
                0.5 * len(indices))))  # flip horizontally 50%
            images[sampled] = np.fliplr(images[sampled])
            sampled = random.sample(sampled, int(round(
                0.25 * len(sampled))))  # flip vertically 25% from above
            images[sampled] = np.flipud(images[sampled])
            return np.array([
                shift(img, [random.randint(-2, 2),
                            random.randint(-2, 2), 0]) for img in images
            ])  # random shift
        else:
            return np.array([
                np.array(
                    self.rand_augment(Image.fromarray(
                        np.reshape(img, self.shape)),
                                      M=random.randint(2, 5)))
                for img in images
            ])

    def agreement_based_labeling(self, y_pre, y_preds=None):
        y_pseudo = np.array(y_pre)
        num = self.num_classes

        y_vote = np.eye(num, np.argmax(y_pseudo, axis=1))
        y_votes = np.sum(
            [np.eye(num, np.argmax(y_rm, axis=1)) for y_rm in y_preds], axis=0)
        y_vote = np.sum([y_vote, y_votes], axis=0)
        y_pseudo = np.eye(num, np.argmax(y_vote, axis=1))

        return y_pseudo
