"""
A federated semi-supervised learning trainer using FedMatch.

Reference:

Jeong et al., "Federated Semi-supervised learning with inter-client consistency and
disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf
"""

import logging
from math import fabs
import os
import random

import numpy as np
from numpy.random.mtrand import f
import torch
from PIL import Image
from scipy.ndimage.interpolation import shift
from torch import autograd, nn, optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from plato.config import Config
from plato.models import de_lenet5_decomposed
from plato.trainers import basic


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
        self.confident = 0.75  #0.75
        self.lambda_s = 10  # supervised learning
        self.lambda_a = 1e-2  # agreement-based pseudo labeling
        self.lambda_i = 1e-2  # inter-client consistency
        self.lambda_l1 = 1e-4
        self.lambda_l2 = 10
        self.helpers = None
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

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                #_train_loader = getattr(self, "train_loader", None)
                """obtain labeled and unlabeled dataset"""

                trainset_s, trainset_u = torch.utils.data.random_split(
                    trainset, [40000, 20000])  # rewrite with ratio
                train_loader_s = DataLoader(trainset_s, batch_size=batch_size)
                """
                train_loader_s = torch.utils.data.DataLoader(
                    dataset=trainset,
                    shuffle=False,
                    batch_size=batch_size,
                    sampler=sampler.get())
                """
                train_loader_u = DataLoader(trainset_u, batch_size)

                #iterations_per_epoch = np.ceil(len(trainset_s) /
                #                               batch_size).astype(int)
                # variable iterations_per_epoch is computed for lr_scheduler;
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

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

                optimizer_s = optim.SGD(
                    #[list(self.model.parameters())[2]],  # 3 is sigma
                    self.model.parameters(),
                    lr=Config().trainer.learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config().trainer.weight_decay
                )  #get_optimizer(self.model.psis)

                optimizer_u = optim.SGD(
                    self.model.parameters(),  #4 is psi
                    lr=Config().trainer.learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config().trainer.weight_decay
                )  #get_optimizer(self.model.sigmas)
                """
                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None
                """

                for epoch in range(1, epochs + 1):
                    self.confident = 0

                    # freeze psi & weight for supervised learning
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and 'weight' in name:
                            param.requires_grad = False
                        if param.requires_grad and 'psi' in name:
                            param.requires_grad = False
                        if param.requires_grad and "sigma" in name:
                            param.requires_grad = True

                    print("=========Supervised Training==========")
                    for batch_id, (examples, labels) in enumerate(
                            train_loader_s):  # batch_id is used for logging
                        #######################
                        # supervised learning
                        #######################
                        with autograd.detect_anomaly():
                            examples, labels = examples.to(
                                self.device), labels.to(self.device)
                            optimizer_s.zero_grad()

                            if cut_layer is None:
                                outputs_s = self.model(examples)
                            else:
                                outputs_s = self.model.forward_from(
                                    examples, cut_layer)

                            loss_s = loss_criterion_s(outputs_s,
                                                      labels)  #* self.lambda_s

                            loss_s.backward()

                            optimizer_s.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader_s),
                                            loss_s.data.item()))
                            else:
                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader_s),
                                            loss_s.data.item()))

                        #######################
                        # unsupervised learning
                        #######################
                    print("=========Unsupervised Training==========")
                    # freeze sigma & weight for supervised learning
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and 'sigma' in name:
                            param.requires_grad = False
                        if not param.requires_grad and 'psi' in name:
                            param.requires_grad = True

                    for batch_id, (examples_unlabeled,
                                   labels) in enumerate(train_loader_u):
                        with autograd.detect_anomaly():
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

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader_u),
                                            loss_u.data.item()))
                            else:
                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader_u),
                                            loss_u.data.item()))

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        self.model.cpu()

        model_type = config['model_name']
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
        self.save_model(filename)

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
            y_pred = self.model(
                unlabled_samples)  #(self.scale(unlabled_samples))
        else:
            y_pred = self.model.forward_from(
                unlabled_samples, cut_layer)  #self.scale(unlabled_samples),
            #cut_layer)
        pred_prob = nn.Softmax(dim=1)
        y_pred = pred_prob(y_pred)

        _confident = np.where(
            np.max(y_pred.detach().numpy(), axis=1) >= self.confident)[0]

        if len(_confident):
            print("_condfident is : ", _confident)

        if len(_confident) > 0:
            # Inter-client consistency
            samples_confident = unlabled_samples[
                _confident]  #self.scale(unlabled_samples[_confident])
            #y_pred = torch.gather(y_pred, 1, _confident)

            y_pred = torch.index_select(y_pred, 0,
                                        torch.from_numpy(_confident))

            if self.helpers is not None:
                print("The type of rm is: ", type(self.helpers[0]))
                y_preds = []
                rm_model = de_lenet5_decomposed.Model()  #self.model
                for rm in self.helpers:
                    rm_model.load_state_dict(rm, strict=True)
                    rm_pred = pred_prob(rm_model(samples_confident))
                    y_preds.append(rm_pred)

                #inter-client consistency loss
                for _, pred in enumerate(y_preds):
                    loss_u += (nn.functional.kl_div(
                        pred, y_pred, reduction='batchmean') /
                               len(y_preds)) * self.lambda_i

            else:
                y_preds = None

            # Agreement-based Pseudo Labeling
            if cut_layer is None:
                y_hard = self.model(
                    unlabled_samples[_confident]
                )  #self.scale(unlabled_samples[_confident]))
                #self.augment(unlabled_samples[_confident],
                #soft=False)))
            else:
                y_hard = self.model.forward_from(unlabled_samples[_confident])
                #self.scale(unlabled_samples[_confident]))
                #self.augment(unlabled_samples[_confident],
                #soft=False)), cut_layer)

            y_pseu = torch.from_numpy(
                self.agreement_based_labeling(y_pred, y_preds))
            loss_u += loss_criterion_u(y_pseu, y_hard) * self.lambda_a

        # Regularization
        self.psi = self.model.get_psi()
        self.sigma = self.model.get_sigma()

        for lid, psi in enumerate(self.psi):  #
            # l1 regularization
            loss_u += torch.sum(torch.abs(psi)) * self.lambda_l1

            # l2 regularization
            loss_u += torch.sum(torch.square(
                (self.sigma[lid] - psi))) * self.lambda_l2

        return loss_u, len(_confident)

    def scale(self, x):
        x = x.numpy() / 255  #astype(np.float32) / 255
        return x

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

        return np.array([
            np.array(
                RandAugment(Image.fromarray(np.reshape(img, self.shape)),
                            M=random.randint(2, 5))) for img in images
        ])

    def agreement_based_labeling(self, y_pre, y_preds_tensor=None):

        y_pseudo = y_pre.detach().numpy()  #np.array(y_pre)

        num = 10  #self.num_classes

        if y_preds_tensor is not None:
            y_preds = []
            for y in y_preds_tensor:
                y_array = y.detach().numpy()
                y_preds.append(y_array)

            y_vote = np.eye(num)[np.argmax(y_pseudo, axis=1)]
            y_votes = np.sum(
                [np.eye(num)[np.argmax(y_rm, axis=1)] for y_rm in y_preds],
                axis=0)
            y_vote = np.sum([y_vote, y_votes], axis=0)
            y_pseudo = np.eye(num)[np.argmax(y_vote, axis=1)]
        #else:
        #np.eye(num)[np.argmax(y_pseudo, axis=1)]

        return y_pseudo
